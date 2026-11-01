"""Diagnostic fit witnesses, never robot task credit.

Run scripts/check_all_scenes.py --driver native --tasks t23 --sequence
configs/t23_fit_probe_20261101.json --gpu DEVICE --output OUTPUT.
"""
from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from wia.runtime.scene_capture_queue import write_json
from wia.tasks.interaction_setup import array


def run(session, output, spec):
    """Place released objects above openings and measure native settling.

    A successful witness proves a tested placement exists. A failed drop does
    not prove the container is unsuitable under every pose or insertion path.
    Geometry and thresholds are unchanged; pose setters are explicitly used.
    """
    import torch
    from scipy.spatial.transform import Rotation
    from omnigibson.utils.usd_utils import PoseAPI
    from wia.authoring.converters.rigid_partial_insertion import RigidPartialInsertionEvaluator

    output = Path(output)
    report = {'schema': 'wia_rigid_fit_probe_v1', 'benchmark_score': False, 'model_calls': 0,
        'probe_revision': ('principal_axis_placement_v8' if spec.get('principal_axes')
                           else 'per_link_collision_geometry_open_hands_v7'),
        'state_setters_used': True, 'robot_task_success_claim': False,
        'scope': 'Existence of stable released partial-insertion witnesses; unsuccessful probes are inconclusive.',
        'specification': deepcopy(spec), 'trials': [], 'errors': [], 'completed': False}
    started = time.monotonic()

    def save():
        report['elapsed_s'] = time.monotonic() - started
        write_json(output/'fit_probe.json', report)

    source = session.observation_source
    source.begin_episode(session.public_task_text)
    native = session.task_evaluator.native
    if not isinstance(native, RigidPartialInsertionEvaluator):
        raise ValueError('Fit probes require the native partial-insertion evaluator.')
    original_parameters = native.p
    original_active = native.active
    targets, containers = spec.get('targets', list(native.p['targets'])), spec['containers']
    if not set(targets) <= set(native.p['targets']) or not set(containers) <= set(native.p['containers']):
        raise ValueError('Fit probe identities must belong to the loaded task.')
    objects = {key: native.objects[key] for key in [*native.p['targets'], *native.p['containers']]}
    original_poses = {key: tuple(array(x).copy() for x in obj.get_position_orientation()) for key, obj in objects.items()}
    original_bounds = {key: tuple(array(v).copy() for v in obj.aabb) for key, obj in objects.items()}
    # Derive world geometry from collision vertices and native body poses,
    # without reading the stale Fabric transform after a root setter. Some
    # targets (e.g. a spray nozzle) have joints, so retain per-link geometry.
    geometry = {}
    initial_joints = {}
    for key, obj in objects.items():
        geometry[key] = [(link, array(link.collision_boundary_points_local)
                          * array(link.get_world_scale()))
                         for link in obj.links.values()
                         if link.collision_boundary_points_local is not None]
        if obj.n_joints > 0:
            initial_joints[key] = array(obj.get_joint_positions()).copy()

    def collision_points(key):
        rows = []
        for link, points in geometry[key]:
            position, quaternion = (array(v) for v in link.get_position_orientation())
            rows.append(Rotation.from_quat(quaternion).apply(points) + position)
        return np.concatenate(rows)

    local_points = {key: Rotation.from_quat(original_poses[key][1]).inv().apply(
        collision_points(key) - original_poses[key][0]) for key in targets}
    report['initial_joint_positions'] = {key: values.tolist() for key, values in initial_joints.items()}
    report['initial_poses'] = {key: [v.tolist() for v in values] for key, values in original_poses.items()}
    report['placement_geometry'] = 'Native per-link collision vertices with measured world scale and body poses; original joint positions restored between drops. Checked against native AABBs after the first physics step.'
    # This only validates the diagnostic placement geometry, not insertion,
    # penetration or stability. It is 5% of the 1 cm release clearance.
    report['placement_readback_tolerance_m'] = 0.0005
    report['thresholds'] = {key: native.p[key] for key in (
        'stability_duration_s', 'maximum_linear_speed_m_s', 'maximum_angular_speed_rad_s',
        'interior_margin_m', 'penetration_tolerance_m', 'withdrawal_clearance_m',
        'container_pose_tolerance_m', 'maximum_container_rotation_rad', 'max_sample_gap_s')}
    report['labels'] = {key: original_parameters['target_labels'][key] for key in targets}
    report['container_dimensions_m'] = {key: (array(native.objects[key].aabb[1])-array(native.objects[key].aabb[0])).tolist() for key in containers}

    def pose(obj, position, quaternion):
        reference = obj.get_position_orientation()[0]
        tensor = lambda x: torch.as_tensor(x, dtype=reference.dtype, device=reference.device)
        obj.set_position_orientation(tensor(position), tensor(quaternion))
        obj.set_linear_velocity(tensor([0., 0., 0.]))
        obj.set_angular_velocity(tensor([0., 0., 0.]))
        obj.wake()
        # Native cached transforms/AABBs can otherwise describe the previous
        # trial after a setter, producing an incorrect drop position.
        PoseAPI.invalidate()

    def step():
        session.step_control(native.controller.hold_action.copy())

    save()
    # Suppress benchmark accumulation in this deliberately state-set diagnostic.
    native.active = False
    try:
        report['preparation_actions'] = []
        for command in ('RELEASE', 'RIGHT_RELEASE'):
            result = native.controller.execute(command)
            report['preparation_actions'].append({'action': command,
                'completed': result['completed'], 'reason': result.get('reason')})
            save()
            if not result['completed']:
                raise ValueError('Native fit preparation could not open both hands.')
        for target_key in targets:
            orientations = [(angles, Rotation.from_euler('xyz', angles, degrees=True).as_quat(), None)
                            for angles in spec['orientations_xyz_deg']]
            if spec.get('principal_axes'):
                eigenvalues, eigenvectors = np.linalg.eigh(np.cov(local_points[target_key], rowvar=False))
                for axis_index in spec['principal_axes']:
                    axis = eigenvectors[:, axis_index]
                    for direction in (1, -1):
                        rotation = Rotation.align_vectors([[0., 0., float(direction)]], [axis])[0]
                        orientations.append((None, rotation.as_quat(), {
                            'axis_index_ascending_variance': axis_index, 'axis_local': axis.tolist(),
                            'eigenvalues': eigenvalues.tolist(), 'vertical_direction': direction}))
            for container_key in containers:
                for angles, quat, principal_axis in orientations:
                    for key, obj in objects.items():
                        if key in initial_joints:
                            positions = obj.get_joint_positions()
                            values = torch.as_tensor(initial_joints[key], dtype=positions.dtype, device=positions.device)
                            obj.set_joint_positions(values)
                            obj.set_joint_velocities(torch.zeros_like(values))
                        pose(obj, *original_poses[key])
                    target, container = objects[target_key], objects[container_key]
                    native.p = dict(original_parameters, targets=[target_key], containers=[container_key])
                    native._contact_witnesses.clear()
                    native._contact_retention_diagnostics.clear()
                    low, high = original_bounds[container_key]
                    rotated = Rotation.from_quat(quat).apply(local_points[target_key])
                    t_low, t_high = rotated.min(axis=0), rotated.max(axis=0)
                    root = (low + high - t_low - t_high) / 2
                    root[2] = high[2] + spec['drop_clearance_m'] - t_low[2]
                    planned_low, planned_high = t_low + root, t_high + root
                    pose(target, root, quat)
                    readback_position, readback_quaternion = (array(v).copy() for v in target.get_position_orientation())
                    trial = {'target': target_key, 'container': container_key,
                        'orientation_xyz_deg': angles, 'initial_position_m': root.tolist(),
                        'principal_axis_orientation': principal_axis,
                        'readback_pose': [readback_position.tolist(), readback_quaternion.tolist()],
                        'planned_initial_target_aabb': [planned_low.tolist(), planned_high.tolist()],
                        'initial_container_aabb': [low.tolist(), high.tolist()],
                        'samples': [], 'stable_witness': False}
                    report['trials'].append(trial)
                    trial['drop_setup_valid'] = bool(
                        np.allclose(((planned_low+planned_high)/2)[:2], ((low+high)/2)[:2], atol=1e-4, rtol=0)
                        and abs(planned_low[2] - high[2] - spec['drop_clearance_m']) <= 1e-4
                        and np.allclose(readback_position, root, atol=1e-4, rtol=0)
                        and np.allclose(Rotation.from_quat(readback_quaternion).as_matrix(),
                                        Rotation.from_quat(quat).as_matrix(), atol=1e-4, rtol=0))
                    save()
                    if not trial['drop_setup_valid']:
                        raise ValueError('Native drop pose does not match the specified opening and clearance.')
                    begin = float(session.pipeline.og.sim.current_time)
                    stable_start = previous_time = None
                    while float(session.pipeline.og.sim.current_time) - begin < spec['maximum_settle_s']:
                        step()
                        if not trial['samples']:
                            points = collision_points(target_key)
                            predicted = np.stack((points.min(axis=0), points.max(axis=0)))
                            measured = np.stack([array(v) for v in target.aabb])
                            trial['post_step_geometry'] = {'predicted_aabb': predicted.tolist(),
                                'native_aabb': measured.tolist(),
                                'maximum_error_m': float(np.max(np.abs(predicted-measured))),
                                'consistent': bool(np.allclose(predicted, measured,
                                    atol=report['placement_readback_tolerance_m'], rtol=0))}
                            save()
                            if not trial['post_step_geometry']['consistent']:
                                raise ValueError('Post-step native collision bounds disagree with rigid placement geometry.')
                        sample = native._measure()
                        now = sample['native_time_s']
                        row = sample['targets'][target_key]
                        relation = row['containers'][container_key]
                        valid = (relation.get('partial_inside') is True and relation.get('touching') is True
                            and relation.get('penetration_free') is True and not row['robot_contact']
                            and row['dynamic_collidable'] is True
                            and native.scorer._grasp(sample)[2] is True
                            and sample['hand_clear'] is True and sample['protected_geometry']['unchanged'] is True
                            and row['linear_speed_m_s'] <= native.p['maximum_linear_speed_m_s']
                            and row['angular_speed_rad_s'] <= native.p['maximum_angular_speed_rad_s'])
                        if previous_time is not None and now - previous_time > native.p['max_sample_gap_s'] + 1e-6:
                            stable_start = None
                        stable_start = (now if stable_start is None else stable_start) if valid else None
                        trial['samples'].append(sample)
                        previous_time = now
                        if stable_start is not None and now - stable_start >= native.p['stability_duration_s']:
                            trial.update(stable_witness=True, stable_start_s=stable_start, stable_end_s=now)
                            break
                    trial['final_pose'] = [array(v).tolist() for v in target.get_position_orientation()]
                    trial['elapsed_simulation_s'] = float(session.pipeline.og.sim.current_time)-begin
                    image_dir = output/'fit_images'/f'{target_key}_{container_key}_{len(report["trials"]):02d}'
                    source.capture(action_boundary=True).observation.save(image_dir)
                    trial['observation_directory'] = str(image_dir)
                    save()
                    print(f'[fit-probe] {target_key} {container_key} {angles}: stable_witness={trial["stable_witness"]}', flush=True)
                    if trial['stable_witness']:
                        break
        report.update(completed=True, evaluation_validity='complete')
    except Exception as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
        raise
    finally:
        native.p = original_parameters
        native.active = original_active
        save()
        # The scene-sweep receipt describes probe execution, not benchmark credit.
        write_json(output/'scorer_validation.json', {key: report[key] for key in
            ('schema', 'benchmark_score', 'model_calls', 'errors', 'completed', 'elapsed_s')}
            | {'evaluation_validity': report.get('evaluation_validity', 'unavailable'), 'diagnostic': 'fit_probe'})
    return report
