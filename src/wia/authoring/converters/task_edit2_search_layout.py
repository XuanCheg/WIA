"""Curated source-preserving search layout and read-only native structural checks.

Run: python -B -m pytest -q tests/test_task_edit2_search_runtime.py
Added tabletop coordinates are proposals; native footprint/contact/access evidence
must qualify each exact reset. Visual-only source props have no support-contact or
graspability certificate, even when their AABBs lie above the table surface.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np

from wia.tasks.interaction_setup import array
from wia.authoring.io import digest

LAYOUT_REVISION = 'task_edit2_search_multistation_20261018_v2'
NATIVE_TABLE_LAYOUT_REVISION = 'task_edit2_search_native_table_translation_20261018_v1'
TABLE_CANDIDATE_CAP = 256
PILOT_LIGHTING = {'revision': 'task_edit2_search_lighting_pilot_20261018_v1',
    'phase_duration_s': 10., 'changed_scale': .10,
    'phases': ['normal', 'changed'], 'calibration_status': 'new_pilot_native_luminance_and_feature_review_pending',
    'source_values_reconstructed': False}


def _bbox(row):
    box = array(row['source_specification']['bbox'])
    if box.shape != (2, 3) or np.any(box[1] <= box[0]):
        raise ValueError('Exact source bounding geometry is required for tabletop planning.')
    return box


def _bounds(group):
    value = array(group['source_render_contract']['room_bbox']['expanded_bbox_world_xy'])
    if value.shape != (4,) or np.any(value[2:] <= value[:2]):
        raise ValueError('A curated tabletop needs explicit source room bounds.')
    return value


def _floor(group, provenance):
    payload = next(row['source_payload'] for row in provenance if row['source_identity'] == group['source_identity'])
    render = payload['question_data']['render']
    poses = render.get('camera_poses', [])
    poses = list(poses.values()) if isinstance(poses, dict) else poses
    if not poses:
        raise ValueError('Source camera/floor proposal unavailable.')
    # This generator uses the source's recorded 1.25 m camera height rule. It is
    # not a native floor measurement and is rechecked against real floor bodies.
    return float(poses[0]['position'][2] - 1.25)


def _table(name, room, xy, floor, dimensions, color):
    top = floor + .75
    rows = [{'name': name, 'position': [*xy, top - .025], 'dimensions_m': [*dimensions, .05],
        'category': 'breakfast_table', 'rgba': color}]
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        height = top - .05 - floor
        rows.append({'name': f'{name}_leg_{i}',
            'position': [xy[0] + sx * (dimensions[0] / 2 - .045),
                         xy[1] + sy * (dimensions[1] / 2 - .045), floor + height / 2],
            'dimensions_m': [.045, .045, height], 'category': 'breakfast_table', 'rgba': color})
    return {'name': name, 'room_instance': room, 'centre_xy_m': list(xy), 'surface_z_m': top,
        'dimensions_xy_m': list(dimensions), 'floor_proposal_z_m': floor, 'primitives': rows,
        'native_floor_footprint_and_contact_verified': False}


def _translate(group_rows, centre_xy, surface):
    boxes = np.stack([_bbox(row) for row in group_rows])
    low, high = boxes[:, 0].min(0), boxes[:, 1].max(0)
    delta = np.r_[array(centre_xy) - (low[:2] + high[:2]) / 2, surface - low[2] + .003]
    for row in group_rows:
        row['native']['position'] = (array(row['source_specification']['position']) + delta).tolist()
        row['initialization_translation_m'] = delta.tolist()
    return {'members': [row['native']['name'] for row in group_rows], 'translation_m': delta.tolist(),
        'original_bounds': [low.tolist(), high.tolist()], 'adapted_bounds': [(low + delta).tolist(), (high + delta).tolist()]}


def _ambiguity_pairs(rows, group):
    targets = [r for r in rows if r['role'] == 'targets']
    confusers = [r for r in rows if r['role'] == 'confusers']
    pairs, used = [], set()
    for entry in group['mechanism_source_annotations']['source_entries']:
        proposal = entry['ball_positions'][0]
        confuser = entry['metadata']['confuser_object']
        choose = lambda candidates, point: min(candidates, key=lambda r: float(np.linalg.norm(
            array(r['source_specification'].get('requested_position', r['source_specification']['position'])) - point)))
        target = choose(targets, proposal)
        distractor = choose([r for r in confusers if r['native']['category'] == confuser['category']], confuser['position'])
        if target['native']['name'] in used or distractor['native']['name'] in used:
            raise ValueError('Source entries do not bind distinct exact ambiguity pairs.')
        if any(np.linalg.norm(array(row['source_specification'].get('requested_position', row['source_specification']['position']))
                              - point) > .001 for row, point in ((target, proposal), (distractor, confuser['position']))):
            raise ValueError('Source ambiguity pairing cannot be reconstructed from saved positions.')
        used.update((target['native']['name'], distractor['native']['name']))
        pairs.append([target, distractor])
    if len(pairs) != 4 or used != {r['native']['name'] for r in rows}:
        raise ValueError('This curated layout requires exactly the four audited source pairs.')
    return pairs


def curate_world(world, groups):
    """Change only cross-pair placement and a whole-cluster common translation."""
    plan = {'revision': LAYOUT_REVISION, 'tables': [], 'dependency_groups': [],
        'native_qualified': False, 'source_pose_provenance_preserved': True}
    inventory = world['joint_inventory']
    for group in groups:
        rows = [r for r in inventory if r['source_identity'] == group['source_identity']]
        room = next(row['source_payload']['room'] for row in world['source_provenance']
                    if row['source_identity'] == group['source_identity'])
        for row in rows:
            row['source_room_instance'] = room
            row['native']['fixed_base'] = True
        if group['mechanism'] == 'Category Ambiguity':
            bounds, floor = _bounds(group), _floor(group, world['source_provenance'])
            pairs = _ambiguity_pairs(rows, group)
            for station in range(2):
                centre = [float(bounds[0] + (station + .5) * (bounds[2] - bounds[0]) / 2), float((bounds[1] + bounds[3]) / 2)]
                table = _table(f'search_ambiguity_station_{station+1}', room, centre, floor,
                               [1.2, 1.6], ([.7, .35, .12, 1.] if station == 0 else [.18, .45, .7, 1.]))
                plan['tables'].append(table)
                for slot, pair in enumerate(pairs[2*station:2*station+2]):
                    position = [centre[0], centre[1] + (-.4 if slot == 0 else .4)]
                    dependency = _translate(pair, position, table['surface_z_m'])
                    dependency.update(kind='exact_target_confuser_pair', table=table['name'],
                        source_identity=group['source_identity'], room_instance=room)
                    for row in pair:
                        row['table_binding'] = table['name']
                    plan['dependency_groups'].append(dependency)
        elif group['mechanism'] == 'Merged Observation':
            entries = group['mechanism_source_annotations']['source_entries']
            if len(entries) != 1 or len(rows) != 4:
                raise ValueError('The audited garage sample must retain its one four-target cluster.')
            bounds, floor = _bounds(group), _floor(group, world['source_provenance'])
            centre = ((bounds[:2] + bounds[2:]) / 2).tolist()
            table = _table('search_garage_cluster_table', room, centre, floor, [.8, .8], [.3, .5, .3, 1.])
            plan['tables'].append(table)
            dependency = _translate(rows, centre, table['surface_z_m'])
            dependency.update(kind='unchanged_compact_cluster', table=table['name'],
                source_identity=group['source_identity'], room_instance=room)
            for row in rows:
                row['table_binding'] = table['name']
            plan['dependency_groups'].append(dependency)
        elif group['mechanism'] == 'Illumination Variability':
            group['lighting_schedule'] = deepcopy(PILOT_LIGHTING)
        else:
            plan['dependency_groups'].append({'kind': 'unchanged_source_room_geometry',
                'members': [r['native']['name'] for r in rows], 'translation_m': [0., 0., 0.],
                'source_identity': group['source_identity'], 'room_instance': room})
    world['curated_layout'] = plan
    return plan


def primitive_configs(world):
    return [{'type': 'PrimitiveObject', 'primitive_type': 'Cube', 'size': 1.,
        'name': row['name'], 'category': row['category'], 'position': row['position'],
        'orientation': [0., 0., 0., 1.], 'scale': row['dimensions_m'], 'rgba': row['rgba'],
        'fixed_base': True, 'visual_only': False}
        for table in world.get('curated_layout', {}).get('tables', []) for row in table['primitives']]


def pose(obj):
    p, q = (array(v) for v in obj.get_position_orientation())
    lo, hi = (array(v) for v in obj.aabb)
    scale = array(obj.scale)
    if p.shape != (3,) or q.shape != (4,) or np.linalg.norm(q) < 1e-8 or lo.shape != (3,) or hi.shape != (3,) or np.any(hi <= lo):
        raise ValueError('Native search pose/bounds unavailable.')
    return {'position': p.tolist(), 'orientation': q.tolist(), 'scale': scale.tolist(),
        'aabb': [lo.tolist(), hi.tolist()], 'visual_only': bool(obj.visual_only),
        'fixed_base': bool(obj.fixed_base), 'category': obj.category, 'model': getattr(obj, 'model', None)}


def same_pose(row, expected, tolerance=.025):
    first, second = array(row['orientation']), array(expected.get('orientation', expected.get('quaternion_xyzw')))
    return bool(np.linalg.norm(array(row['position']) - expected['position']) <= tolerance
                and min(np.linalg.norm(first - second), np.linalg.norm(first + second)) <= .025)



def _native_floor_ray(start, end):
    from omnigibson.utils.sampling_utils import raytest
    return raytest(start, end, only_closest=False)


def geometric_floor_foot_support(foot, floors, assembly_paths):
    """Measure a fixed foot against native floor shapes; this is not Touching."""
    low, high = (array(v) for v in foot.aabb)
    floor_paths = [obj.prim_path for obj in floors]
    under = lambda body,root: str(body)==root or str(body).startswith(root+'/')
    maximum_gap_m, maximum_numerical_penetration_m = .006, .002
    samples = []
    for x,y in [(float((low[0]+high[0])/2),float((low[1]+high[1])/2))] + [
            (float(x),float(y)) for x in (low[0]+.001,high[0]-.001) for y in (low[1]+.001,high[1]-.001)]:
        hits = _native_floor_ray([x,y,float(low[2])+.05],[x,y,float(low[2])-.05])
        floors_hit, foreign = [], []
        for hit in hits:
            if not isinstance(hit,dict) or not hit.get('hit'):
                continue
            body = hit.get('rigidBody','')
            if any(under(body,root) for root in assembly_paths):
                continue
            if any(under(body,root) for root in floor_paths) and float(array(hit['normal'])[2])>=.9:
                floors_hit.append(hit)
            else:
                foreign.append({'rigid_body':str(body),'position':array(hit['position']).tolist()})
        if floors_hit:
            hit = max(floors_hit,key=lambda row:float(array(row['position'])[2]))
            gap = float(low[2]-array(hit['position'])[2])
            passed = -maximum_numerical_penetration_m<=gap<=maximum_gap_m and not foreign
            row = {'xy_m':[x,y],'floor_prim_path':str(hit['rigidBody']),
                'floor_surface_position_m':array(hit['position']).tolist(),
                'floor_surface_normal':array(hit['normal']).tolist(),'bottom_to_floor_gap_m':gap,
                'foreign_obstructions':foreign,'accepted':bool(passed)}
        else:
            row = {'xy_m':[x,y],'accepted':False,'reason':'native_floor_shape_hit_missing',
                   'foreign_obstructions':foreign}
        samples.append(row)
    contacts = list(foot.contact_list())
    contact_value = None if not contacts else any(any(under(str(body),root) for root in floor_paths)
        for contact in contacts for body in (contact.body0,contact.body1))
    return {'geometric_floor_foot_support':bool(samples and all(row['accepted'] for row in samples)),
        'method':'native_leg_bottom_AABB_and_vertical_floor_shape_rays',
        'maximum_gap_m':maximum_gap_m,'maximum_numerical_penetration_m':maximum_numerical_penetration_m,
        'samples':samples,'physical_floor_contact_stream':contact_value,
        'touching_state_claimed':False}


def resolve_runtime_world(session, original):
    """Use one accepted private layout only for its original contract and reset."""
    runtime = getattr(session, '_search_world_contract', None)
    binding = getattr(session, '_search_world_contract_binding', None)
    if runtime is None and binding is None:
        return original, None
    expected = {'original_world_digest': digest(original), 'reset_id': getattr(session, 'native_reset_id', None)}
    if (not isinstance(runtime, dict) or not isinstance(binding, dict) or expected['reset_id'] is None
            or any(binding.get(key) != value for key, value in expected.items())
            or binding.get('runtime_layout_digest') != digest(runtime)):
        raise ValueError('Search runtime layout belongs to another original world/reset or changed after freezing.')
    receipt = getattr(session, '_search_table_layout_initialization', None)
    if (not isinstance(receipt, dict) or receipt.get('context') != expected
            or receipt.get('runtime_layout_digest') != binding['runtime_layout_digest']):
        raise ValueError('Search runtime layout has no matching native initialization receipt.')
    return runtime, receipt


def _invalidate_native_pose_cache(session):
    from omnigibson.utils.usd_utils import PoseAPI
    PoseAPI.invalidate()
    invalidate = getattr(session, 'invalidate_camera_cache', None)
    if callable(invalidate):
        invalidate()


def _set_assembly(session, objects, baseline, delta):
    import torch
    for name, obj in objects.items():
        obj.set_position_orientation(position=torch.as_tensor(array(baseline[name]['position']) + delta,
            dtype=torch.float32, device='cpu'), orientation=torch.as_tensor(baseline[name]['orientation'],
            dtype=torch.float32, device='cpu'))
    _invalidate_native_pose_cache(session)


def _native_table_cells(session, room, original_xy, coverage):
    """Enumerate raw, uneroded native floor cells by exact deterministic distance."""
    import torch
    trav = session.env.scene.trav_map
    chunks = []
    for floor_index, floor_map in enumerate(trav.floor_map):
        raw = array(floor_map)
        if raw.ndim != 2:
            raise ValueError('Raw native floor traversability cells are unavailable.')
        cells = np.argwhere(raw == 255)
        if not len(cells):
            continue
        xy = array(trav.map_to_world(torch.as_tensor(cells, device='cpu')))
        if xy.shape != (len(cells), 2):
            raise ValueError('Native traversability map-to-world coordinates are unavailable.')
        hints = getattr(trav, 'floor_heights', None)
        hint = float(hints[floor_index]) if hints is not None else None
        if hint is not None and not math.isfinite(hint):
            raise ValueError('Native floor-map height disambiguation is unavailable.')
        chunks.append((floor_index, cells, xy, hint))
    coverage['raw_native_cell_count'] = sum(len(row[1]) for row in chunks)
    coverage['map_resolution_m'] = float(trav.map_resolution)
    coverage['raw_cell_rule'] = 'native_floor_map_equals_255_without_erosion'
    candidates = [(float(np.sum((point - original_xy) ** 2)), floor_index, int(cell[0]), int(cell[1]), point, hint)
                  for floor_index, cells, points, hint in chunks for cell, point in zip(cells, points)]
    candidates.sort(key=lambda row: row[:4])
    query = session.env.scene.seg_map.get_room_instance_by_point
    for _, floor_index, row, col, point, hint in candidates:
        coverage['raw_cells_scanned'] += 1
        if query(point) != room:
            continue
        coverage['exact_room_candidates_encountered'] += 1
        yield point, hint, {'floor_index': floor_index, 'native_cell_rc': [row, col]}
    coverage['raw_map_exhausted'] = True


def _floor_plane(table, original_rows, xy_delta, floors, assembly_paths, hint):
    top = array(original_rows[table['name']]['aabb'])
    low, high = top[0, :2] + xy_delta, top[1, :2] + xy_delta
    sites = [(float(x), float(y)) for x in np.linspace(low[0] + .01, high[0] - .01, 3)
             for y in np.linspace(low[1] + .01, high[1] - .01, 3)]
    for name, row in original_rows.items():
        if '_leg_' not in name:
            continue
        box = array(row['aabb'])
        sites.extend((float(x), float(y)) for x in (box[0, 0] + .001 + xy_delta[0], box[1, 0] - .001 + xy_delta[0])
                     for y in (box[0, 1] + .001 + xy_delta[1], box[1, 1] - .001 + xy_delta[1]))
        sites.append(tuple(((box[0, :2] + box[1, :2]) / 2 + xy_delta).tolist()))
    floor_paths = [str(obj.prim_path) for obj in floors]
    lower = min(float(array(obj.aabb[0])[2]) for obj in floors) - .1
    upper = max(float(array(obj.aabb[1])[2]) for obj in floors) + .1
    under = lambda body, root: str(body) == root or str(body).startswith(root + '/')
    samples, heights, normals = [], [], []
    reference = table['floor_proposal_z_m'] if hint is None else hint
    for x, y in sites:
        hits = _native_floor_ray([x, y, upper], [x, y, lower])
        measured = []
        foreign = []
        for hit in hits:
            if not isinstance(hit, dict) or hit.get('hit') is not True:
                continue
            body = str(hit.get('rigidBody', ''))
            if any(under(body, root) for root in assembly_paths):
                continue
            point, normal = array(hit['position']), array(hit['normal'])
            if any(under(body, root) for root in floor_paths):
                if normal.shape == (3,) and np.linalg.norm(normal[:2]) <= .01 and normal[2] >= .99:
                    measured.append({'rigid_body': body, 'position_m': point.tolist(), 'normal': normal.tolist()})
            else:
                foreign.append({'rigid_body': body, 'position_m': point.tolist()})
        if not measured:
            samples.append({'xy_m': [x, y], 'accepted': False, 'reason': 'horizontal_native_floor_shape_hit_missing',
                            'foreign_hits': foreign})
            continue
        selected = min(measured, key=lambda row: (abs(row['position_m'][2] - reference), row['rigid_body']))
        heights.append(selected['position_m'][2]); normals.append(selected['normal'])
        samples.append({'xy_m': [x, y], 'accepted': True, 'selected_floor_hit': selected, 'foreign_hits': foreign})
    coplanar = bool(len(heights) == len(sites) and max(heights) - min(heights) <= .002
                    and np.max(np.linalg.norm(array(normals) - normals[0], axis=1)) <= .01)
    return {'accepted': coplanar, 'surface_z_m': max(heights) if coplanar else None,
        'samples': samples, 'coplanarity_tolerance_m': .002, 'normal_difference_tolerance': .01,
        'floor_hint_role': 'hit_disambiguation_only_not_floor_evidence', 'floor_hint_z_m': hint,
        'floor_identity_source': 'actual_loaded_collidable_floor_shape_rays'}


def _proposal_clock(session):
    reset = getattr(session, 'native_reset_id', None)
    time = float(session.pipeline.og.sim.current_time)
    if reset is None or not math.isfinite(time):
        raise ValueError('Exact native reset/time required for table proposal bounds snapshot.')
    return str(reset), time


def _other_native_bounds(session, all_actors, assembly_names):
    values = session.env.scene.objects
    values = values.values() if isinstance(values, dict) else values
    for other in values:
        if other.name in assembly_names or other.category in ('floors', 'floor'):
            continue
        if other.visual_only and other.name not in all_actors:
            continue
        low, high = (array(value) for value in other.aabb)
        yield {'name': other.name, 'category': other.category, 'aabb_m': [low.tolist(), high.tolist()],
               'visual_only': bool(other.visual_only)}


@dataclass(frozen=True)
class _ProposalBoundsSnapshot:
    """Immutable numeric other-actor bounds, scoped to one paused table search."""
    reset_id: str
    native_time_s: float
    all_actors: frozenset
    assembly_names: frozenset
    inventory_names: tuple
    bounds: tuple

    def validate(self, session, all_actors, assembly_names):
        if (_proposal_clock(session) != (self.reset_id, self.native_time_s)
                or frozenset(all_actors) != self.all_actors or frozenset(assembly_names) != self.assembly_names):
            raise ValueError('Table proposal bounds snapshot reset/time or actor exclusions changed.')

    def rows(self):
        return [{'name': name, 'category': category, 'visual_only': visual,
                 'aabb_m': [list(low), list(high)]} for name, category, visual, low, high in self.bounds]

    def report(self):
        eligible = {row[0] for row in self.bounds}
        return {'schema': 'search_initial_table_proposal_bounds_snapshot_v1',
            'reset_id': self.reset_id, 'native_time_s': self.native_time_s,
            'inventory_names': list(self.inventory_names), 'eligible_names': [row[0] for row in self.bounds],
            'excluded_names': [name for name in self.inventory_names if name not in eligible],
            'assembly_names': sorted(self.assembly_names), 'all_task_actor_names': sorted(self.all_actors),
            'native_AABB_measurement_read_count': len(self.bounds), 'bounds': self.rows(),
            'scope': 'initialization_preplacement_rejection_only', 'immutable_numeric_snapshot': True,
            'postplacement_acceptance_uses_fresh_native_bounds': True}


def _snapshot_proposal_bounds(session, all_actors, assembly_names):
    reset, time = _proposal_clock(session)
    values = session.env.scene.objects
    values = list(values.values()) if isinstance(values, dict) else list(values)
    rows = tuple((row['name'], row['category'], row['visual_only'],
                  tuple(row['aabb_m'][0]), tuple(row['aabb_m'][1]))
                 for row in _other_native_bounds(session, all_actors, assembly_names))
    snapshot = _ProposalBoundsSnapshot(reset, time, frozenset(all_actors), frozenset(assembly_names),
                                       tuple(obj.name for obj in values), rows)
    snapshot.validate(session, all_actors, assembly_names)
    return snapshot


def _table_screen(session, table, rows, delta, all_actors, assembly_names, floor_z, *, other_bounds_snapshot=None):
    if other_bounds_snapshot is not None:
        other_bounds_snapshot.validate(session, all_actors, assembly_names)
    scene = session.env.scene
    boxes = {name: array(row['aabb']) + delta for name, row in rows.items()}
    top = boxes[table['name']]
    query = scene.seg_map.get_room_instance_by_point
    room_queries = [{'xy_m': [float(x), float(y)], 'room_instance': query(np.array([x, y]))}
        for x in np.linspace(top[0, 0] + .01, top[1, 0] - .01, 3)
        for y in np.linspace(top[0, 1] + .01, top[1, 1] - .01, 3)]
    checks = {'exact_room_footprint': all(row['room_instance'] == table['room_instance'] for row in room_queries)}
    assigned = [name for name in rows if name not in {primitive['name'] for primitive in table['primitives']}]
    checks['complete_props_fit_table_bounds'] = all(np.all(boxes[name][0, :2] >= top[0, :2] + .005)
        and np.all(boxes[name][1, :2] <= top[1, :2] - .005) and boxes[name][0, 2] >= top[1, 2] - .005
        and boxes[name][0, 2] <= top[1, 2] + .08 for name in assigned)
    upper = max(float(box[1, 2]) for box in boxes.values())
    obstacles = []
    bounds = (other_bounds_snapshot.rows() if other_bounds_snapshot is not None else
              _other_native_bounds(session, all_actors, assembly_names))
    for other in bounds:
        low, high = (array(value) for value in other['aabb_m'])
        if (np.all(high[:2] > top[0, :2] + .002) and np.all(low[:2] < top[1, :2] - .002)
                and high[2] > floor_z + .025 and low[2] < upper + .002):
            obstacles.append({'name': other['name'], 'category': other['category'], 'aabb_m': [low.tolist(), high.tolist()],
                              'visual_only': other['visual_only']})
    checks['clear_other_native_bounds'] = not obstacles
    if other_bounds_snapshot is not None:
        other_bounds_snapshot.validate(session, all_actors, assembly_names)
    return {'accepted': all(checks.values()), 'checks': checks, 'room_queries': room_queries,
        'native_bounds_obstacles': obstacles, 'collision_screen_height_m': [floor_z + .025, upper + .002],
        'proposed_assembly_bounds': {name: box.tolist() for name, box in boxes.items()}}


def initialize_search_tables(session, original):
    """Translate only declared table assemblies using measured native room/floor data."""
    # Source postprocessing precedes session.robot/controller assignment. Reading
    # the lazy observation_source property here would create a rig or raise.
    state = vars(session)
    sources = [state.get(name) for name in ('_observation_source', 'observation_source')]
    observed = any(source is not None and (any(getattr(source, name, None) is not None for name in
        ('initial_observation', 'latest_snapshot', 'boundary_observation'))
        or getattr(source, 'action_step', 0) or getattr(source, 'history', ())) for source in sources)
    controllers = [state.get('controller'), state.get('task_controller'),
        getattr(state.get('task_evaluator'), 'controller', None)]
    controllers.extend(getattr(getattr(source, 'controller_state_provider', None), '__self__', None)
                       for source in sources if source is not None)
    acted = any(controller is not None and (getattr(controller, '_public_action_sequence', 0)
        or getattr(controller, '_public_action_context', None)) for controller in [session, *controllers])
    if state.get('_search_joint_postprocessed', False) or observed or acted:
        raise ValueError('Measured Search table relocation is initialization-only, before the first observation.')
    if getattr(session, '_search_table_layout_initialization', None) is not None:
        raise ValueError('Search table initialization already has a retained receipt; replay is forbidden.')
    runtime = deepcopy(original)
    context = {'original_world_digest': digest(original), 'reset_id': getattr(session, 'native_reset_id', None)}
    receipt = {'revision': NATIVE_TABLE_LAYOUT_REVISION, 'context': context, 'applicable': False,
        'accepted': True, 'groups': [], 'first_failure': None, 'candidate_cap_per_table': TABLE_CANDIDATE_CAP,
        'physics_steps': 0, 'background_modified': False, 'robot_start_modified': False,
        'access_qualified': False, 'informative_images_qualified': False,
        'visual_props_support_contact_claimed': False, 'graspability_claimed': False}
    session._search_table_layout_initialization = receipt
    scene = session.env.scene
    registry = lambda name: scene.object_registry('name', name)
    all_objects = list(scene.objects.values() if isinstance(scene.objects, dict) else scene.objects)
    floors = [obj for obj in all_objects if obj.category in ('floor', 'floors') and not obj.visual_only]
    all_actors = {row['native']['name'] for row in original['joint_inventory']} | {row['name'] for row in primitive_configs(original)}
    for table in runtime.get('curated_layout', {}).get('tables', []):
        receipt['applicable'] = True
        assigned = [row for row in runtime['joint_inventory'] if row.get('table_binding') == table['name']]
        declared = [group for group in runtime['curated_layout']['dependency_groups'] if group.get('table') == table['name']]
        names = {row['native']['name'] for row in assigned} | {row['name'] for row in table['primitives']}
        result = {'table': table['name'], 'room_instance': table['room_instance'], 'accepted': False,
            'declared_members': sorted(row['native']['name'] for row in assigned), 'attempts': [],
            'coverage': {'raw_cells_scanned': 0, 'exact_room_candidates_encountered': 0, 'raw_map_exhausted': False,
                         'candidate_cap': TABLE_CANDIDATE_CAP, 'failure_certifies_impossibility': False}}
        receipt['groups'].append(result)
        baseline = None
        moved = False
        try:
            if context['reset_id'] is None or not floors:
                raise ValueError('Exact reset and actual collidable floor inventory are required.')
            members = [name for group in declared for name in group['members']]
            if not assigned or len(members) != len(set(members)) or set(members) != {row['native']['name'] for row in assigned}:
                raise ValueError('The table must own exactly its declared source dependency groups.')
            objects = {name: registry(name) for name in names}
            if any(obj is None for obj in objects.values()):
                raise ValueError('A declared table/prop assembly member is missing.')
            baseline = {name: pose(obj) for name, obj in objects.items()}
            result['original_proposed_native_poses'] = deepcopy(baseline)
            protected = {obj.name: pose(obj) for obj in all_objects if obj.name not in names}
            proposal_bounds = _snapshot_proposal_bounds(session, all_actors, names)
            result['proposal_bounds_snapshot'] = proposal_bounds.report()
            paths = [str(obj.prim_path) for obj in objects.values()]
            original_xy = array(table['centre_xy_m'])
            candidates = iter(_native_table_cells(session, table['room_instance'], original_xy, result['coverage']))
            candidate = (original_xy, None, {'kind': 'original_proposal'})
            for index in range(TABLE_CANDIDATE_CAP):
                xy, floor_hint, origin = candidate
                horizontal = array(xy) - original_xy
                floor = _floor_plane(table, baseline, horizontal, floors, paths, floor_hint)
                delta = np.r_[horizontal, 0. if index == 0 else floor['surface_z_m'] - table['floor_proposal_z_m']
                              if floor['accepted'] else 0.]
                trial = {'candidate_index': index, 'origin': origin, 'candidate_xy_m': array(xy).tolist(),
                    'common_delta_m': delta.tolist(), 'floor_shape_measurement': floor, 'accepted': False}
                result['attempts'].append(trial)
                if not floor['accepted']:
                    trial['failure_stage'] = 'native_horizontal_coplanar_floor_shape_measurement'
                else:
                    screen = _table_screen(session, table, baseline, delta, all_actors, names, floor['surface_z_m'],
                                           other_bounds_snapshot=proposal_bounds)
                    trial['screen'] = screen
                    if not screen['accepted']:
                        trial['failure_stage'] = next(key for key, value in screen['checks'].items() if not value)
                    else:
                        if np.any(delta != 0):
                            moved = True
                            _set_assembly(session, objects, baseline, delta)
                        actual = {name: pose(obj) for name, obj in objects.items()}
                        final = _table_screen(session, table, actual, np.zeros(3), all_actors, names, floor['surface_z_m'])
                        trial['actual_native_screen'] = final
                        feet, foreign = {}, {}
                        floor_paths = [str(obj.prim_path) for obj in floors]
                        for primitive in table['primitives']:
                            obj = objects[primitive['name']]
                            foreign[obj.name] = sorted({str(body) for contact in obj.contact_list() for body in (contact.body0, contact.body1)
                                if not any(str(body) == root or str(body).startswith(root + '/') for root in paths + floor_paths)})
                            if '_leg_' in obj.name:
                                feet[obj.name] = geometric_floor_foot_support(obj, floors, paths)
                        trial['geometric_floor_foot_support'] = feet
                        trial['foreign_contact_bodies'] = foreign
                        preserved = all(same_pose(pose(registry(name)), row, .000001) and np.allclose(pose(registry(name))['scale'], row['scale'], atol=1e-6, rtol=0)
                                        for name, row in protected.items())
                        trial['protected_native_objects_unchanged'] = preserved
                        trial['accepted'] = bool(final['accepted'] and feet and all(row['geometric_floor_foot_support'] for row in feet.values())
                                                 and not any(foreign.values()) and preserved)
                        if trial['accepted']:
                            result.update(accepted=True, accepted_delta_m=delta.tolist(), accepted_candidate_index=index,
                                accepted_native_poses=actual, original_proposal=deepcopy(table))
                            for primitive in table['primitives']:
                                primitive['position'] = (array(primitive['position']) + delta).tolist()
                            table['centre_xy_m'] = (array(table['centre_xy_m']) + delta[:2]).tolist()
                            table['surface_z_m'] += float(delta[2]); table['floor_proposal_z_m'] += float(delta[2])
                            table['native_floor_footprint_and_contact_verified'] = False
                            table['native_geometric_floor_foot_support_verified'] = True
                            for row in assigned:
                                row['native']['position'] = (array(row['native']['position']) + delta).tolist()
                                row['initialization_translation_m'] = (array(row['initialization_translation_m']) + delta).tolist()
                            for group in declared:
                                group['translation_m'] = (array(group['translation_m']) + delta).tolist()
                                group['adapted_bounds'] = (array(group['adapted_bounds']) + delta).tolist()
                            break
                        trial['failure_stage'] = ('protected_native_objects_changed' if not preserved else
                            'actual_table_bounds_floor_feet_or_foreign_contact')
                        if moved:
                            _set_assembly(session, objects, baseline, np.zeros(3)); moved = False
                if result.get('first_rejection') is None:
                    result['first_rejection'] = deepcopy(trial)
                if index + 1 >= TABLE_CANDIDATE_CAP:
                    break
                try:
                    xy, hint, origin = next(candidates)
                    candidate = (xy, hint, {'kind': 'raw_native_traversability_cell', **origin})
                except StopIteration:
                    break
            if not result['accepted']:
                result['failure_reason'] = 'No accepted table assembly placement in the bounded measured native candidate search.'
        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError, ImportError, IndexError) as error:
            result['failure_reason'] = f'{type(error).__name__}: {error}'
        finally:
            if not result['accepted'] and moved and baseline is not None:
                _set_assembly(session, objects, baseline, np.zeros(3))
        if not result['accepted']:
            receipt['accepted'] = False
            if receipt['first_failure'] is None:
                receipt['first_failure'] = deepcopy(result)
    if receipt['applicable']:
        runtime['curated_layout']['runtime_layout_revision'] = NATIVE_TABLE_LAYOUT_REVISION
    receipt['runtime_layout_digest'] = digest(runtime)
    return runtime, receipt

def native_structure(session, world):
    """Measure exact loaded actors, floor/footprint, tables and unchanged furniture."""
    original_digest = digest(world)
    scene = session.env.scene
    objects = scene.objects.values() if isinstance(scene.objects, dict) else scene.objects
    objects = list(objects)
    rows, checks, errors = {}, {}, {}
    geometric_foot_support = {}
    registry = lambda name: scene.object_registry('name', name)
    try:
        from wia.tasks.native_floor_contacts import measure_native_floor_contacts
        world, layout_receipt = resolve_runtime_world(session, world)
        checks['runtime_layout_context_bound'] = layout_receipt is not None
        if world.get('curated_layout', {}).get('tables'):
            checks['native_table_layout_initialization'] = bool(layout_receipt and layout_receipt.get('accepted') is True)
        checks['scene_model'] = session.scene_name == world['scene_model']
        floors = [o for o in objects if o.category in ('floors', 'floor') and not o.visual_only]
        if not floors:
            raise ValueError('Loaded native collidable floor bodies unavailable.')
        floor_paths = [obj.prim_path for obj in floors]
        robot = getattr(session, 'robot', None)
        if robot is None:
            robot = session.env.robots[0]
        robot_state = pose(robot)
        contacts = measure_native_floor_contacts(session, robot, floor_paths)
        q = array(robot_state['orientation']); q /= np.linalg.norm(q)
        checks['floor_supported_start'] = (bool(contacts['floor_supported'] is True and contacts['collision_free'] is True
            and 1 - 2*(q[0]**2 + q[1]**2) >= .95) if contacts.get('measurement_available') is True else None)
        checks['floor_bodies_loaded'] = True
        query = scene.seg_map.get_room_instance_by_point
        loaded_rooms = {room for obj in objects for room in (getattr(obj, 'in_rooms', None) or ())}
        checks['loaded_required_rooms'] = set(world['required_rooms']).issubset(loaded_rooms)
        names = set()
        for item in world['joint_inventory']:
            definition = item['native']; name = definition['name']; names.add(name)
            obj = registry(name)
            if obj is None:
                raise ValueError('Native joint inventory actor absent: ' + name)
            rows[name] = row = pose(obj)
            row['measured_room'] = query(array(row['position'])[:2])
            checks['identity:' + name] = obj.name == name and all(row[field] == definition[field] for field in ('category', 'model'))
            checks['placement:' + name] = same_pose(row, definition) and row['visual_only'] and row['fixed_base']
            checks['room:' + name] = row['measured_room'] == item['source_room_instance']
            expected_dimensions = _bbox(item)[1] - _bbox(item)[0]
            checks['dimensions:' + name] = bool(np.max(np.abs(array(row['aabb'])[1] - array(row['aabb'])[0] - expected_dimensions)) <= .03)
            baseline = getattr(session, '_search_initial_native', {}).get(name)
            reference_scale = definition.get('scale', baseline.get('scale') if baseline else None)
            checks['scale:' + name] = reference_scale is not None and bool(np.allclose(row['scale'], reference_scale, atol=1e-6, rtol=0))
        names.update(row['name'] for row in primitive_configs(world))
        for table in world.get('curated_layout', {}).get('tables', []):
            top = registry(table['name'])
            if top is None:
                raise ValueError('A curated table is absent: ' + table['name'])
            footprint_low, footprint_high = array(top.aabb[0]), array(top.aabb[1])
            sample_rooms = [query(np.array([x, y])) for x in np.linspace(footprint_low[0]+.01, footprint_high[0]-.01, 3)
                            for y in np.linspace(footprint_low[1]+.01, footprint_high[1]-.01, 3)]
            checks['table_room_footprint:' + top.name] = all(room == table['room_instance'] for room in sample_rooms)
            assembly = {p['name'] for p in table['primitives']} | {item['native']['name'] for item in world['joint_inventory']
                if item.get('table_binding') == table['name']}
            screen_rows = {name: pose(registry(name)) for name in assembly}
            screen = _table_screen(session, table, screen_rows, np.zeros(3), names, assembly, table['floor_proposal_z_m'])
            # Conservative loaded native AABB exclusion; actual static contacts
            # are reported independently and do not replace this footprint check.
            checks['table_clear_native_furniture_bounds:' + top.name] = screen['checks']['clear_other_native_bounds']

            for primitive in table['primitives']:
                obj = registry(primitive['name'])
                row = pose(obj); rows[obj.name] = row; names.add(obj.name)
                checks['table_geometry:' + obj.name] = (same_pose(row, {'position': primitive['position'], 'orientation': [0.,0.,0.,1.]})
                    and not row['visual_only'] and row['fixed_base']
                    and np.allclose(row['scale'], primitive['dimensions_m'], atol=1e-6, rtol=0))
                allowed_paths = [registry(n).prim_path for n in assembly]
                foreign = []
                for contact in obj.contact_list():
                    for body in (str(contact.body0), str(contact.body1)):
                        if not any(body == root or body.startswith(root + '/') for root in allowed_paths + floor_paths):
                            foreign.append(body)
                checks['table_no_foreign_collision:' + obj.name] = not foreign
                if '_leg_' in obj.name:
                    receipt = geometric_floor_foot_support(obj, floors, allowed_paths)
                    geometric_foot_support[obj.name] = receipt
                    checks['geometric_floor_foot_support:' + obj.name] = receipt['geometric_floor_foot_support']
            for item in world['joint_inventory']:
                if item.get('table_binding') != table['name']:
                    continue
                box = array(rows[item['native']['name']]['aabb'])
                checks['visual_table_bounds:' + item['native']['name']] = bool(
                    np.all(box[0,:2] >= footprint_low[:2]+.005) and np.all(box[1,:2] <= footprint_high[:2]-.005)
                    and box[0,2] >= footprint_high[2]-.005 and box[0,2] <= footprint_high[2]+.08)
        for dependency in world.get('curated_layout', {}).get('dependency_groups', []):
            members = dependency['members']
            source = {item['native']['name']: item['source_specification'] for item in world['joint_inventory'] if item['native']['name'] in members}
            for name in members[1:]:
                actual = array(rows[name]['position']) - rows[members[0]]['position']
                expected = array(source[name]['position']) - source[members[0]]['position']
                checks['dependency_vector:' + name] = bool(np.linalg.norm(actual-expected) <= .004)
        for group in world.get('search_groups', []):
            for entry in group.get('mechanism_source_annotations', {}).get('source_entries', []):
                anchor = entry.get('metadata', {}).get('anchor_object')
                if not anchor:
                    continue
                obj = registry(anchor['name'])
                if obj is None:
                    raise ValueError('Source occlusion/boundary anchor is absent: ' + anchor['name'])
                actual = pose(obj)
                checks['anchor_geometry:' + obj.name] = (obj.category == anchor['category'] and
                    np.max(np.abs(array(actual['aabb']) - array(anchor['bbox']))) <= .05)
        baseline = getattr(session, '_search_background_initial', None)
        if baseline is None:
            checks['background_unchanged'] = False
        else:
            checks['background_unchanged'] = all(registry(name) is not None and same_pose(pose(registry(name)), row, .01)
                                                for name, row in baseline.items())
        # Target bounds/raycast geometry are visual placement measurements only.
        return {'available': True, 'checks': {k: bool(v) if v is not None else None for k,v in checks.items()}, 'objects': rows,
            'robot_contacts': contacts, 'geometric_floor_foot_support': geometric_foot_support, 'loaded_rooms': sorted(loaded_rooms), 'floor_names': [o.name for o in floors],
            'source_relations_preserved': all(v for k,v in checks.items() if k.startswith(('dependency_vector:', 'placement:', 'dimensions:', 'scale:', 'anchor_geometry:'))),
            'target_support_contact_claimed': False, 'graspability_claimed': False,
            'world_digest': original_digest, 'runtime_layout_digest': digest(world),
            'native_table_layout_initialization': layout_receipt, 'measurement_errors': errors}
    except (ValueError, TypeError, AttributeError, KeyError, RuntimeError, IndexError, ImportError) as error:
        return {'available': False, 'checks': checks, 'objects': rows, 'measurement_errors': {'native_structure': f'{type(error).__name__}: {error}'},
            'target_support_contact_claimed': False, 'graspability_claimed': False, 'world_digest': original_digest}


class SearchLightCycle:
    """Reuse the native source light reader/writer, sampling the actual sim clock."""
    def __init__(self, session, lamps, lights):
        from .enumerative_perception_illumination_variability import _OriginalLightCycle
        self.session, self.lamps = session, lamps
        self.native = _OriginalLightCycle(session, deepcopy(PILOT_LIGHTING), lights)
        self.last_time = float(session.pipeline.og.sim.current_time)
        self.samples = 0
        self.restored = False

    def sample(self, *_):
        if self.restored:
            return
        now = float(self.session.pipeline.og.sim.current_time)
        delta = now - self.last_time
        if not math.isfinite(now) or delta < -1e-8:
            self.native.error = 'Native simulation clock is discontinuous.'
            return
        if delta > 0:
            self.native.advance(delta)
            self.samples += 1
            self.last_time = now

    def measure(self):
        self.sample()
        return {**self.native.measure(), 'pilot_schedule': deepcopy(PILOT_LIGHTING),
            'elapsed_simulation_s': self.native.elapsed_s, 'physics_time_s': self.last_time,
            'unique_clock_samples': self.samples, 'lamp_bindings': self.lamps,
            'cycle_running': not self.restored}

    def restore(self):
        self.restored = True
        for light in self.native.lights:
            light['attribute'].Set(light['baseline'])
        return {'restored_original_intensities': True,
                'native_light_measurement': self.native.measure()}
