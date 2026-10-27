"""Measure controlled ramp trials followed by an unused-spare delivery.

Run generated candidates with scripts/check_esi_core_native.py --evaluate.
This evaluator reads native contacts, motion and time; it never moves objects.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math

import numpy as np
from scipy.spatial.transform import Rotation

from wia.authoring.converters.physical_dynamics_inclined_plane import _array, _pose, _native_time, _friction
from wia.authoring.io import write_json
from wia.tasks.interaction_setup import array as native_array

MODULE = 'core.esi_converters.inclined_comparison'


def qualified_fastest(records, candidates, tolerance_s=.15):
    """Only complete controlled trials can define an application choice."""
    if any(key not in records or not records[key]['valid'] for key in candidates):
        return []
    times = {key: records[key]['arrival_s'] for key in candidates
             if records[key]['arrival_s'] is not None and records[key].get('final_in_receiver') is True}
    if not times:
        return []
    best = min(times.values())
    return [key for key, seconds in times.items() if seconds <= best + tolerance_s]


def _under(path, root):
    return str(path) == str(root) or str(path).startswith(str(root) + '/')


def qualify_operation(session, controller, config, bound):
    parameters = config['parameters']
    checks = {'supported_native_layout': getattr(session, 'category_interaction_setup', {}).get('accepted') is True}
    try:
        for pair in parameters['candidate_pairs']:
            trial, spare = bound[pair['trial']], bound[pair['spare']]
            checks['matched_spare:' + pair['trial']] = (type(trial) is type(spare)
                and trial.category == spare.category and getattr(trial, 'model', None) == getattr(spare, 'model', None)
                and np.allclose(_array(trial.scale), _array(spare.scale)) and
                getattr(trial, '_primitive_type', None) == getattr(spare, '_primitive_type', None) and
                (getattr(trial, '_primitive_type', None) != 'Sphere' or abs(trial.radius - spare.radius) < 1e-6) and
                not trial.fixed_base and not spare.fixed_base and not trial.visual_only and not spare.visual_only)
        material = _friction(bound['ramp'])
        checks['ramp_friction'] = bool(material and all(
            abs(row['static_friction'] - parameters['static_friction']) < 1e-4 and
            abs(row['dynamic_friction'] - parameters['dynamic_friction']) < 1e-4 for row in material))
        return {'accepted': all(checks.values()), 'checks': checks, 'ramp_materials': material,
            'controlled_trials_and_application_verified': False}
    except (ValueError, AttributeError, TypeError) as error:
        return {'accepted': False, 'checks': checks, 'error': str(error)}


class InclinedComparisonEvaluator:
    def __init__(self, session, controller, config, output):
        self.session, self.controller, self.config = session, controller, config
        self.p, self.output = config['parameters'], Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.objects = {key: session.env.scene.object_registry('name', name) for key, name in config['bindings'].items()}
        self.trials = [row['trial'] for row in self.p['candidate_pairs']]
        self.spares = {row['spare']: row['trial'] for row in self.p['candidate_pairs']}
        self.active = False
        self.records, self.traces, self.held_ready, self.receiver_since = {}, {}, {}, {}
        self.last_held, self.selected, self.frozen = None, None, None
        self.constraints_ok, self.application_valid = True, False
        self.last_clock = None

    def begin_episode(self, observation):
        qualification = qualify_operation(self.session, self.controller, self.config, self.objects)
        if not qualification['accepted']:
            raise ValueError('Ramp comparison initialization failed: ' + str(qualification))
        self.initial = {key: _pose(obj) for key, obj in self.objects.items()}
        self.ramp_materials = _friction(self.objects['ramp'])
        self.active = True
        return self.record(observation)

    def _measure(self, key):
        from omnigibson.object_states import Touching
        obj, ramp, receiver = self.objects[key], self.objects['ramp'], self.objects['receiver']
        row = _pose(obj)
        centre = (np.array(row['aabb_min']) + row['aabb_max']) / 2
        ramp_position, ramp_quaternion = ( _array(x) for x in ramp.get_position_orientation())
        rotation = Rotation.from_quat(ramp_quaternion)
        local = rotation.inv().apply(centre - ramp_position)
        half = np.array(self.p['ramp_dimensions_m']) / 2
        top = False
        interference = []
        allowed = [self.objects[name].prim_path for name in self.p['trial_support_keys']]
        for contact in obj.contact_list():
            external = [str(body) for body in (contact.body0, contact.body1) if not _under(body, obj.prim_path)]
            if not external:
                continue
            if all(_under(body, ramp.prim_path) for body in external):
                location = rotation.inv().apply(native_array(contact.position) - ramp_position)
                normal = rotation.inv().apply(native_array(contact.normal))
                top |= bool(abs(location[2] - half[2]) <= .01 and abs(normal[2]) >= .5)
            if any(not any(_under(body, support) for support in allowed) for body in external):
                interference.extend(external)
        rlo, rhi = (_array(x) for x in receiver.aabb)
        in_receiver = (np.all(np.array(row['aabb_min'])[:2] >= rlo[:2] - .003) and
            np.all(np.array(row['aabb_max'])[:2] <= rhi[:2] + .003) and
            bool(obj.states[Touching].get_value(receiver)))
        reference_key = self.spares.get(key, key)
        reference_q = np.array(self.initial[reference_key]['quaternion_xyzw'])
        q = np.array(row['quaternion_xyzw'])
        angle = 2 * math.acos(min(1., abs(float(np.dot(q / np.linalg.norm(q), reference_q)))))
        orientation_ok = key in self.p.get('orientation_free_keys', []) or obj.category == 'baseball' or angle <= math.radians(20.)
        release_ready = (top and abs(local[0] - self.p['release_local_x_m']) <= .025 and
            abs(local[1]) <= .025 and orientation_ok and
            row['linear_speed_m_s'] <= .05 and row['angular_speed_rad_s'] <= .2)
        dlo, dhi = self.p['safe_bounds_m']
        safe = (np.all(np.array(row['aabb_min'])[:2] >= np.array(dlo)[:2] - .01) and
            np.all(np.array(row['aabb_max'])[:2] <= np.array(dhi)[:2] + .01) and
            row['aabb_min'][2] >= dlo[2] - .04)
        return {**row, 'release_ready': bool(release_ready), 'in_receiver': bool(in_receiver),
            'safe': bool(safe), 'interference': interference, 'local_ramp_position': local.tolist()}

    def _sample(self):
        if not self.active:
            return
        now = _native_time()['time_s']
        if now == self.last_clock:
            return
        if self.last_clock is not None and now < self.last_clock:
            raise ValueError('Native clock moved backward.')
        self.last_clock = now
        control = self.controller.geometric_rearrangement_evidence()
        if control.get('available') is not True:
            raise ValueError('Native public-control evidence is unavailable.')
        self.constraints_ok &= control.get('state_setter_used') is False
        grasp = self.controller.debug_gripper_state()
        names = {obj.name: key for key, obj in self.objects.items()}
        held = names.get(grasp.get('object_name')) if grasp.get('holding') and grasp.get('constraint_active') else None
        if held is not None and held not in self.trials and held not in self.spares:
            self.constraints_ok = False
        for key in self.p['apparatus_keys']:
            pose = _pose(self.objects[key])
            initial = self.initial[key]
            self.constraints_ok &= bool(np.linalg.norm(np.array(pose['position']) - initial['position']) <= .005 and
                np.allclose(pose['scale'], initial['scale']) and
                abs(float(np.dot(pose['quaternion_xyzw'], initial['quaternion_xyzw']))) >= .9999)
        if held in self.spares and self.frozen is None:
            self.frozen = deepcopy(self.records)
            self.selected = held
            self.application_valid = self.spares[held] in qualified_fastest(
                self.frozen, self.trials, self.p['arrival_tie_tolerance_s'])
        elif held in self.spares and held != self.selected:
            self.application_valid = False
        if self.frozen is not None and held in self.trials:
            self.constraints_ok = False
        for spare in self.spares:
            if spare != self.selected and np.linalg.norm(np.array(_pose(self.objects[spare])['position']) -
                    self.initial[spare]['position']) > .03:
                self.constraints_ok = False
        for key in [*self.trials, *self.spares]:
            row = self._measure(key)
            if held == key:
                self.traces.pop(key, None)
                self.receiver_since.pop(key, None)
                self.held_ready[key] = row['release_ready']
                continue
            if self.last_held == key and grasp.get('release_pending') is False:
                if self.held_ready.get(key):
                    self.traces[key] = {'release_time': now, 'last_time': now, 'samples': 0,
                        'valid': True, 'arrival_s': None, 'complete': False}
                else:
                    self.traces.pop(key, None)
            trace = self.traces.get(key)
            if trace is None:
                continue
            if not trace['complete']:
                trace['valid'] &= bool(now - trace['last_time'] <= .1 and row['safe'] and not row['interference'])
                trace['last_time'], trace['samples'] = now, trace['samples'] + 1
                if row['in_receiver'] and trace['arrival_s'] is None:
                    trace['arrival_s'] = now - trace['release_time']
                if now - trace['release_time'] >= self.p['trial_duration_s']:
                    trace['complete'] = True
                    if key in self.trials:
                        self.records[key] = {**deepcopy(trace), 'valid': bool(trace['valid'] and trace['samples'] >= 30),
                            'final_in_receiver': row['in_receiver']}
            stable = (row['in_receiver'] and not row['interference'] and row['linear_speed_m_s'] <= .02 and
                row['angular_speed_rad_s'] <= .1 and grasp.get('object_name') is None and
                grasp.get('release_pending') is False)
            if stable:
                self.receiver_since.setdefault(key, now)
            else:
                self.receiver_since.pop(key, None)
        # The motor release can span several samples after the constraint clears.
        # Retain its identity until release completion so this edge is not lost.
        if held is not None or grasp.get('release_pending') is False:
            self.last_held = held

    def sample_step(self):
        self._sample()

    def record(self, observation, *, action=None):
        from wia.tasks.completion import progress
        self._sample()
        complete = all(key in self.records and self.records[key]['valid'] for key in self.trials)
        trace = self.traces.get(self.selected, {})
        delivered = bool(self.selected in self.receiver_since and trace.get('complete') and trace.get('valid') and
            self.last_clock - self.receiver_since[self.selected] >= 1.)
        items = [{'id': 'controlled_trials', 'kind': 'checkpoint', 'completed': complete, 'evaluation_validity': 'valid'},
            {'id': 'evidence_based_spare_selection', 'kind': 'checkpoint', 'completed': self.application_valid,
             'evaluation_validity': 'valid'},
            {'id': 'ramp_delivery', 'kind': 'terminal_condition', 'completed': delivered, 'evaluation_validity': 'valid'}]
        value = {'schema': 'inclined_comparison_v1', 'snapshot_id': observation.snapshot_id,
            'action_step': observation.action_step, 'completion_items': items, 'evaluation_validity': 'complete',
            'task_success': bool(self.constraints_ok and complete and self.application_valid and delivered),
            'constraints_satisfied': bool(self.constraints_ok), 'trial_records': self.records,
            'frozen_selection_evidence': self.frozen, 'selected_spare': self.selected,
            'release_traces': self.traces}
        value.update(progress(value))
        write_json(self.output / 'state.json', value)
        return value


def create_evaluator(session, controller, config, output):
    return InclinedComparisonEvaluator(session, controller, config, output)
