"""Check exact authored pickup placement; run pytest on this file with tests on PYTHONPATH."""
from copy import deepcopy
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from wia.authoring.converters import wia_room_collection as collection
from harness_integration.test_wia_room_collection import layout, _initialization_fixture


@pytest.mark.parametrize('drift,accepted', [(0., True), (.014, True), (.016, False)])
@pytest.mark.parametrize('preload_displacement', [(0., 0., 0.), (-.1, .03, -.4)])
def test_authored_xy_survives_release_edge_policy_and_is_measured_after_settling(
        monkeypatch, drift, accepted, preload_displacement):
    session, objects, _ = _initialization_fixture(monkeypatch)
    spec = layout()
    spec['allow_authored_navigation_inventory'] = True
    spec['added_inventory'] = spec.pop('native_inventory')
    spec['native_inventory'] = []
    session._release_edge_policy = True
    for row in spec['added_inventory']:
        obj = objects[row['name']]
        row.update(position=obj.get_position_orientation()[0].tolist(), orientation=[0., 0., 0., 1.],
                   pickup_xy_policy='preserve_authored_v1', maximum_pickup_xy_drift_m=.015)
        obj.position = np.array(row['position'])
        obj.get_position_orientation = lambda obj=obj: (obj.position.copy(), np.array([0., 0., 0., 1.]))
        def set_pose(position, orientation, obj=obj):
            delta = np.array(position) - obj.position
            obj.aabb = [value + delta for value in obj.aabb]
            obj.position = np.array(position)
        obj.set_position_orientation = set_pose
        obj.keep_still = lambda: None
        obj.set_position_orientation(position=obj.position + preload_displacement,
                                     orientation=[0., 0., 0., 1.])
    def forbidden(*args, **kwargs):
        raise AssertionError('An explicitly preserved pickup XY must not be moved by the edge policy.')
    monkeypatch.setattr('wia.tasks.pickup_edges.apply_edge_policy', forbidden)
    def settle(*args):
        objects['a'].position[0] += drift
        return {'accepted': True}
    monkeypatch.setitem(sys.modules, 'wia.runtime.sim.esi_collection_setup', SimpleNamespace(initialize_collection_tray=settle))
    result = collection.initialize_scene(session, {'layout': spec, 'operation': collection.ORDERED_OPERATION})
    assert result['accepted'] is accepted
    assert result['inventory']['a']['authored_xy_drift_m'] == pytest.approx(drift)
    assert all(row['accepted'] for row in result['inventory_supports'].values())
    assert len(result['support_alignments']) == 2
    assert all(row['authored_pose_restored_before_support_check']
               for row in result['pickup_edge_placements'])


@pytest.mark.parametrize('changes', [{'pickup_xy_policy': 'unknown'}, {'maximum_pickup_xy_drift_m': .1},
    {'pickup_edge_margin_m': .03}, {'pickup_approach_xy_m': [0., 0.]}])
def test_authored_xy_cannot_silently_conflict_with_other_placement_policies(changes):
    spec = layout()
    spec['allow_authored_navigation_inventory'] = True
    row = deepcopy(spec['native_inventory'].pop())
    row.update(position=[2., 0., 1.05], orientation=[0., 0., 0., 1.],
               pickup_xy_policy='preserve_authored_v1', maximum_pickup_xy_drift_m=.015)
    row.update(changes)
    spec['added_inventory'] = [row]
    with pytest.raises(ValueError, match='Preserved pickup XY'):
        collection.validate_layout(spec, collection.ORDERED_OPERATION)
