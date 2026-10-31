"""Estimate efficient decisions including grasp/placement, then apply a 1.3 cap.

Run in the simulator Python environment (CPU only):
  python scripts/estimate_decision_budgets.py --output outputs/geometric_budget_estimate_20261031
No simulator/model call or task/config modification is performed. Private geometry
is used only offline and must never be provided to a policy as an observation.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
DETOUR = 1.4
MARGIN = 1.3


def l1(a, b):
    return sum(abs(float(x)-float(y)) for x, y in zip(a[:2], b[:2]))


def open_route(start, points, ordered=False):
    orders = [points] if ordered else itertools.permutations(points)
    return min(([start, *order] for order in orders),
               key=lambda route: sum(l1(a, b) for a, b in zip(route, route[1:])))


def recipe(transfers=0, travel=2., turns=4, inspect=3, extra=0, reports=1,
           transfer_steps=18, note='Local travel is an explicit engineering assumption.'):
    return dict(transfers=transfers, assumed_travel_m=travel, turn_steps=turns,
                inspection_steps=inspect, extra_steps=extra, report_steps=reports,
                steps_per_transfer=transfer_steps, basis=note)


def mapping_route(definition, alias):
    p = definition['private']['evaluation']['parameters']
    if alias == 'T04':
        from wia.tasks.loader import load_task
        source = json.loads(load_task(ROOT/'explore_to_achieve/wia_v1/release/tasks/t04.yaml',
                                      run_mode='diagnostic').source_path.read_text())
        q = json.loads(source['metadata_json'])['question_data']
        points = q['path_views']['path_world_xy']
        # This source path is straight; samples are evidence locations, not mandatory stops.
        assert all(abs(float(v[1])-float(points[0][1])) < 1e-5 for v in points)
        return [q['initial_view']['camera_pose']['position'][:2], points[0], points[-1]], len(q['required_regions'])
    rooms = {}
    for member in p['members']:
        source = json.loads((ROOT/'ESI/dataset/json_clean'/member['source_identity']).read_text())
        q = json.loads(source['metadata_json'])['question_data']
        for row in q['room_views']:
            if row['room_instance'] in q['required_regions']:
                rooms.setdefault(row['room_instance'], row['center_xy'])
    start = p['mapping_joint_world_contract']['common_start_proposal']['position'][:2]
    return open_route(start, list(rooms.values())), len(rooms)


def estimates():
    primitives = json.loads((ROOT/'configs/esi_primitives_fine.json').read_text())
    limits = json.loads((ROOT/'configs/esi_parameterized_fine.json').read_text())
    base_step = min(limits['max_translation_m'], primitives['base']['translation_m']*limits['max_primitives'])
    arm_step = min(limits['max_translation_m'], primitives['arm']['translation_m']*limits['max_arm_primitives'])
    # One ordinary pick/place: approach .24, descend .16, GRASP, lift .16,
    # carry .40 (or .56 to the other-hand tray), lower .16, RELEASE, withdraw .16.
    def transfer(carry):
        return sum(math.ceil((d-1e-10)/arm_step) for d in (.24, .16, .16, carry, .16, .16)) + 2
    ordinary, tray = transfer(.40), transfer(.56)
    assert (round(base_step, 2), round(arm_step, 2), ordinary, tray) == (.4, .08, 18, 20)
    defaults = {
        'AP-1': recipe(2, extra=4), 'AP-2': recipe(2, extra=6),
        'GF-1': recipe(1, extra=4, inspect=5, reports=2),
        'GF-2': recipe(1, extra=9), 'GF-3': recipe(3, extra=12, inspect=5),
        'GG-2': recipe(1, extra=8), 'GG-3': recipe(1, inspect=5, reports=2),
        'GG-4': recipe(1, extra=2), 'SR-1': recipe(3, travel=6., turns=6, inspect=4, extra=6),
        'SR-2': recipe(2, travel=4., extra=4), 'SR-3': recipe(1, inspect=4, extra=4),
        'TM-1': recipe(3, travel=10., turns=8, inspect=18, reports=2),
        'TM-2': recipe(1, inspect=4, extra=10),
        'NT-3': recipe(travel=6., turns=6, inspect=4, reports=2),
    }
    overrides = {
        'T21': recipe(2, travel=8., turns=6, inspect=4, extra=6),
        'T23': recipe(3, travel=4., extra=8, inspect=6, reports=2),
        'T24': recipe(1, travel=6., turns=6, extra=4, inspect=5, reports=2),
        'T25': recipe(1, travel=6., turns=6, extra=4, inspect=5, reports=2),
        'GF3-2': recipe(2, extra=8, inspect=4),
        'T49': recipe(0, extra=20, note='Cloth sweep 8 plus target grasp/lift/withdraw 12; target need not be delivered.'),
        'T52': recipe(0, travel=0., turns=0, inspect=2),
        'R-S3': recipe(2, inspect=4, extra=6),
        'T10': recipe(travel=12., turns=8, inspect=6, reports=2),
        'T11': recipe(travel=8., turns=6, inspect=4, reports=2),
        'iw_h1_02': recipe(2, inspect=4, extra=4, note='Two transfers cover quarantine/replacement; add four underside inspection/rotation actions.'),
        'iw_h2_02': recipe(3, inspect=6, note='Obstruction storage, payload delivery, one disturbance repair.'),
        'iw_h3_03': recipe(1, inspect=6, note='One deficit refill after the revision event, plus inventory checks.'),
        'iw_h4_02': recipe(3, inspect=6, note='Assume two of three blockers must move, plus target; third blocker would add 18 efficient steps.'),
        'iw_h5_02': recipe(1, inspect=4),
        'iw_m1_01': recipe(2, inspect=6),
        'iw_m2_01': recipe(3, inspect=6, note='Three cartons suffice: 2+2 for order A, 3 for order B; leave quantity-1 carton untouched.'),
        'iw_m3_01': recipe(1, inspect=6, extra=16, note='Add four fridge open/close operations at four actions each.'),
        'iw_m4_02': recipe(6, travel=6., turns=8, inspect=6, extra=12,
                          note='Three jobs each move to station and delivery; twelve additional time/inspection actions are an assumption, not a feasible schedule proof.'),
    }
    catalogue = json.loads((ROOT/'explore_to_achieve/wia_v1/release/catalogue.json').read_text())['tasks']
    result = []
    for entry in catalogue:
        alias, subclass = entry['alias'], entry['subclass']
        if alias in ('T42', 'T43', 'T44'):
            result.append(dict(alias=alias, subclass=subclass, excluded=True, basis='Excluded by user; existing budget unchanged.'))
            continue
        d = yaml.safe_load((ROOT/entry['task_file']).read_text())
        p = d['private']['evaluation']['parameters']
        spec = dict(overrides.get(alias, defaults.get(subclass, {})))
        route = None
        if alias in ('T07', 'T08', 'T35', 'T36'):
            layout = p['layout']
            targets = [r for r in layout['added_inventory'] if r.get('role') == 'target']
            points = [r.get('pickup_approach_xy_m', r['position'][:2]) for r in targets]
            route = open_route(layout['robot_start']['position'][:2], points, ordered=alias in ('T07', 'T08'))
            spec = recipe(len(targets), travel=0., turns=2*len(targets), inspect=len(targets), transfer_steps=tray,
                          note='Authored start/target or pickup-approach XY proxy; ordered for T07/T08, shortest open L1 tour for T35/T36; not a collision-free base path.')
        elif alias in ('T04', 'T05', 'T06'):
            route, rooms = mapping_route(d, alias)
            spec = recipe(travel=0., turns=2*rooms, inspect=4*rooms, reports=2 if alias=='T04' else 6,
                          note='Saved source path for T04; shortest open L1 tour over distinct required-room centers for T05/T06. Centers are proxies, not native traversal certificates.')
        if not spec:
            raise ValueError('Missing estimate recipe: ' + alias)
        if route is None:
            distance = spec['assumed_travel_m']
            nav = math.ceil(DETOUR*distance/base_step)
            route_basis = 'assumed_local_travel'
        else:
            distance = sum(l1(a, b) for a, b in zip(route, route[1:]))
            nav = sum(math.ceil(max(0., DETOUR*abs(float(a[k])-float(b[k]))/base_step-1e-9))
                      for a, b in zip(route, route[1:]) for k in (0, 1))
            route_basis = 'extracted_coordinate_proxy'
        handling = spec['transfers']*spec['steps_per_transfer'] + spec['extra_steps']
        optimum = nav + spec['turn_steps'] + spec['inspection_steps'] + handling + spec['report_steps']
        cap = (13*optimum+9)//10
        if alias == 'T52':
            cap = 10  # Explicit user override for the already-contained control.
        result.append(dict(alias=alias, subclass=subclass, excluded=False,
            route_basis=route_basis, l1_route_m=round(distance, 3), detour_factor=DETOUR,
            navigation_steps=nav, turn_steps=spec['turn_steps'], transfers=spec['transfers'],
            steps_per_transfer=spec['steps_per_transfer'], handling_and_special_steps=handling,
            inspection_steps=spec['inspection_steps'], report_steps=spec['report_steps'],
            estimated_optimal_steps=optimum, proposed_cap=cap, basis=spec['basis'], route_xy=route))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = estimates()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'estimates.json').write_text(json.dumps({'method': 'efficient_engineering_estimate_not_proven_optimum',
        'base_max_m': .4, 'arm_max_m': .08, 'ordinary_transfer_steps': 18, 'tray_transfer_steps': 20,
        'final_cap_multiplier': MARGIN, 'explicit_cap_overrides': {'T52': 10}, 'tasks': rows}, indent=2)+'\n')
    keys = list(dict.fromkeys(k for row in rows for k in row if k != 'route_xy'))
    with (args.output/'estimates.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(row['alias'], row.get('estimated_optimal_steps', 'excluded'), row.get('proposed_cap', 'unchanged'))
    chosen = [r for r in rows if not r['excluded']]
    print('TOTAL', len(chosen), sum(r['estimated_optimal_steps'] for r in chosen), sum(r['proposed_cap'] for r in chosen))


if __name__ == '__main__':
    main()
