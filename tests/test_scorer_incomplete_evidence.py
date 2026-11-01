"""Exercise incomplete goals versus missing measurements. Run: python -m pytest tests/test_scorer_incomplete_evidence.py."""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
from wia.candidates.high import evaluate as high
from wia.candidates.medium import evaluate as medium
from wia.tasks.completion import progress
import test_candidate_high as h
import test_candidate_medium as m
from harness_integration.test_wia_history_collection import replay, obs, receipt
from harness_integration.test_wia_room_collection import episode, finish


@pytest.mark.parametrize('prototype,factory', [('H1',h.h1),('H2',lambda:h.h2(False)),('H3',lambda:h.h3(2)),('H5',lambda:h.h5(False)),('M1',m.m1)])
def test_measured_initial_incomplete_is_numeric(prototype,factory):
    contract,frames=factory()
    value=(high if prototype.startswith('H') else medium)(prototype,contract,frames[0],[])
    assert value['evaluation_validity']=='complete'
    assert value['task_success'] is False
    assert progress(value)['completion_score']==0.


def test_m4_absent_transport_is_false_but_missing_reader_is_unavailable():
    contract,frames=m.timed_trajectory({'a':{'station':'station_a','duration_s':.2}})
    for f in frames:f['objects']['a']['held']=False
    value=medium('M4',contract,frames[-1],frames[:-1])
    assert value['checks']['load_transport:a'] is False
    assert value['checks']['delivery_transport:a'] is False
    assert value['evaluation_validity']=='complete' and not value['task_success']
    frames[2]['objects']['a']['held']=None
    value=medium('M4',contract,frames[-1],frames[:-1])
    assert value['evaluation_validity']=='unavailable'


def test_missing_native_placement_does_not_become_zero():
    contract,frames=h.h1()
    frames[-1]['objects']['package_a'].pop('contacts')
    value=high('H1',contract,frames[-1],frames[:-1])
    assert value['evaluation_validity']=='unavailable' and progress(value)['completion_score'] is None


def test_t38_unopened_query_preserves_survey_quarter(replay):
    evaluator,_,visual,*_=replay
    assert evaluator.record(obs(0))['score_percent']==0.
    for index,key in enumerate('abcdef',1):
        visual['objects']={key:receipt(key,index)}
        result=evaluator.record(obs(index))
    assert result['score_percent']==25. and result['total_goals']==4
    assert result['per_item']=={} and evaluator.physical_available is False
    evaluator.complete_survey(obs(6))
    evaluator._placement=lambda key:{'measurement_error':'missing native reader'}
    result=evaluator.record(obs(7))
    assert result['score_percent'] is None


def test_clock_regression_never_reuses_old_complete_inventory(episode):
    evaluator,current,observation,_=episode
    before=finish(episode)
    assert progress(before)['completion_score']==1.
    current['native_time_s']-=1
    after=evaluator.record(observation)
    assert progress(after)['completion_score'] is None and not after['task_success']
    assert after['clock_regression']['current_time_s']<after['clock_regression']['previous_time_s']
    current['native_time_s']+=100
    assert progress(evaluator.record(observation))['completion_score'] is None


def test_m4_grasp_at_destination_cannot_relabel_push_as_transport():
    contract,frames=m.timed_trajectory({'a':{'station':'station_a','duration_s':.2}})
    # The job is already on the processing station before the first grasp.
    frames[0]['objects']['a']=m.obj(1)
    value=medium('M4',contract,frames[-1],frames[:-1])
    assert value['checks']['load_transport:a'] is False
    assert not value['task_success'] and value['evaluation_validity']=='complete'


@pytest.mark.parametrize('grasp', [None, {}, {'holding': False, 'object_name': None}])
def test_ramp_missing_grasp_telemetry_is_not_a_release(monkeypatch, grasp):
    from wia.authoring.converters import inclined_comparison as ramp
    evaluator = ramp.InclinedComparisonEvaluator.__new__(ramp.InclinedComparisonEvaluator)
    evaluator.active, evaluator.last_clock, evaluator.constraints_ok = True, None, True
    evaluator.controller = NS(debug_gripper_state=lambda: grasp,
        geometric_rearrangement_evidence=lambda: {'available': True, 'state_setter_used': False})
    monkeypatch.setattr(ramp, '_native_time', lambda: {'time_s': 1.})
    with pytest.raises(ValueError, match='grasp history is unavailable'):
        evaluator._sample()


@pytest.mark.parametrize('retrieved', [False, True])
def test_t50_missing_reveal_measurement_cannot_be_hidden_by_valid_placement(tmp_path, retrieved):
    from wia.authoring.converters.deformable_redrape import UncoverSelectPlaceEvaluator
    evaluator = UncoverSelectPlaceEvaluator.__new__(UncoverSelectPlaceEvaluator)
    evaluator.base = NS(measurement_error='cloth reader unavailable', invalid=False, output=tmp_path / 'uncover')
    evaluator.placement = NS(last={'placed': True})
    evaluator.retrieved, evaluator.other_preserved = retrieved, True
    evaluator._sample = lambda: (True, True)
    result = evaluator.record(NS(snapshot_id=1, action_step=0))
    assert result['evaluation_validity'] == ('partial' if retrieved else 'unavailable')
    assert result['completion_score'] is None and not result['task_success']
    assert result['completion_items'][0]['completed'] is retrieved
