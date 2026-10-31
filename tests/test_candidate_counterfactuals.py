"""Check decision changes in the actual candidate catalogue.

Run: PYTHONPATH=src python -m pytest -q tests/test_candidate_counterfactuals.py
These are analytical contracts, not robot execution tests.
"""
import yaml

from wia.resources import ROOT
from wia.candidates.medium import enumerate_allocations, enumerate_schedules


def _contract(identity):
    path=ROOT/'explore_to_achieve/candidates/interactiveworld_20261029/tasks'/(identity+'.yaml')
    return yaml.safe_load(path.read_text())['private']['evaluation']['parameters']['contract']


def test_catalogued_schedule_pair_requires_different_optimal_ordering():
    results=[]
    for identity in ('iw_m4_01','iw_m4_02'):
        contract=_contract(identity)
        results.append(enumerate_schedules(contract['jobs'],contract['handling_s']))
    signatures=[{tuple((op['operation'],op['job']) for op in witness)
                 for witness in result['witnesses']} for result in results]
    assert all(result['enumeration_complete'] for result in results)
    assert [result['optimal_makespan_s'] for result in results]==[24.,28.]
    assert signatures[0] and signatures[1] and signatures[0].isdisjoint(signatures[1])


def test_catalogued_allocation_pair_is_feasible_and_accepts_interchangeable_units():
    results=[]
    for identity in ('iw_m2_01','iw_m2_02'):
        contract=_contract(identity)
        results.append(enumerate_allocations(contract['units'],contract['orders']))
    assert [len(result) for result in results]==[1,3]
    signatures=[{tuple(sorted(assignment.items())) for assignment in result} for result in results]
    assert signatures[0].isdisjoint(signatures[1])
