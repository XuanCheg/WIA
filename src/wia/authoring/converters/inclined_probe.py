"""Initialization-only passive ramp diagnostics; never an Agent success test.

Run through a diagnostic copy of a generated ramp task, replacing its
active_layout hooks with this module and adding evaluation_parameters.
The original native state is restored before capture.
"""
import numpy as np

from wia.authoring.converters import active_layout
from wia.authoring.converters.inclined_comparison import InclinedComparisonEvaluator
from wia.authoring.converters.physical_dynamics_inclined_plane import _array, _pose

configure_scene = active_layout.configure_scene
postprocess_joint_world = active_layout.postprocess_joint_world


def initialize_scene(session, parameters):
    import omnigibson as og
    import torch
    from scipy.spatial.transform import Rotation

    result = active_layout.initialize_scene(session, parameters)
    p = parameters['evaluation_parameters']
    objects = {entry['name']: session.env.scene.object_registry('name', entry['name'])
        for entry in parameters['layout']['objects']}
    evaluator = object.__new__(InclinedComparisonEvaluator)
    evaluator.session, evaluator.p, evaluator.objects = session, p, objects
    evaluator.spares = {pair['spare']: pair['trial'] for pair in p['candidate_pairs']}
    evaluator.initial = {key: _pose(obj) for key, obj in objects.items()}
    baseline = og.sim.dump_state(serialized=False)
    ramp = objects['ramp']
    position, quaternion = (_array(v) for v in ramp.get_position_orientation())
    rotation = Rotation.from_quat(quaternion)
    normal = rotation.as_matrix()[:, 2]
    point = position + rotation.apply([p['release_local_x_m'], 0., p['ramp_dimensions_m'][2] / 2])
    records = {}
    try:
        for pair in p['candidate_pairs']:
            og.sim.load_state(baseline, serialized=False)
            obj = objects[pair['trial']]
            low, high = (_array(v) for v in obj.aabb)
            centre = (low + high) / 2
            support_radius = float(np.abs(normal) @ ((high - low) / 2))
            old_position, orientation = obj.get_position_orientation()
            delta = point + normal * (support_radius + .001) - centre
            obj.set_position_orientation(position=old_position + torch.tensor(delta, dtype=torch.float32),
                orientation=orientation)
            obj.keep_still()
            start = float(og.sim.current_time)
            arrival = None
            rows = []
            while float(og.sim.current_time) - start < p['trial_duration_s']:
                og.sim.step()
                measured = evaluator._measure(pair['trial'])
                elapsed = float(og.sim.current_time) - start
                if measured['in_receiver'] and arrival is None:
                    arrival = elapsed
                rows.append({'time_s': elapsed, 'position_m': measured['position'],
                    'in_receiver': measured['in_receiver'], 'safe': measured['safe'],
                    'interference': measured['interference'],
                    'speed_m_s': measured['linear_speed_m_s']})
            records[pair['trial']] = {'arrival_s': arrival,
                'final_in_receiver': rows[-1]['in_receiver'],
                'valid': all(row['safe'] and not row['interference'] for row in rows),
                'samples': rows, 'release_placement': 'AABB support on the ramp plane, zero velocity, 1 mm clearance'}
    finally:
        og.sim.load_state(baseline, serialized=False)
    session.category_interaction_setup['passive_ramp_probe'] = {
        'records': records, 'initial_native_state_restored': True,
        'robot_grasp_or_public_action_verified': False,
        'scope': 'Passive native physics after initialization-only placement; not an Agent rollout.'}
    return result
