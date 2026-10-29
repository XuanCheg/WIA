#!/usr/bin/env python3
"""Calibrate one ramp material condition with explicit passive object setup.

Run with one selected local CUDA device and isolated OMNIGIBSON_APPDATA_PATH:
  python -B src/wia/runtime/commands/check_wia_ramp_material.py --task-file TASK --scene-file SCENE
    --output-dir FRESH_OUTPUT [--repetitions 3]
    [--release-protocol project_then_refresh|contact_then_zero_velocity]
This is not an Agent run or a lawful robot-release reference. Native full-state
restores and object pose/velocity setters are intentional apparatus calibration.
CPU checks: python -B -m pytest -q -p no:cacheprovider
tests/test_wia_ramp_material_math.py
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = WIA_RUNTIME_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = 'wia_passive_ramp_material_calibration_v1'
SETUP_CLEARANCE_M = .0001
MINIMUM_SAMPLES = 30
MAXIMUM_SAMPLE_GAP_S = .1
RELEASE_PROTOCOLS = ('project_then_refresh', 'contact_then_zero_velocity')
DEFAULT_RELEASE_PROTOCOL = RELEASE_PROTOCOLS[0]
# Readback invariants for the additional passive-calibration velocity reset only.
# These do not replace or relax any evaluator release/geometry/speed threshold.
RELEASE_POSITION_TOLERANCE_M = 1e-7
RELEASE_ORIENTATION_TOLERANCE_RAD = 1e-7
RELEASE_ZERO_VELOCITY_TOLERANCE = 1e-7
RELEASE_CLOCK_TOLERANCE_S = 1e-9
NON_AGENT = {
    'autonomous_agent_performance': False, 'public_controls_only': False,
    'environment_state_setters_used': True, 'legal_robot_reference_verified': False,
    'robot_trial_valid': False, 'model_calls': 0,
}
MATERIAL_BINDING_DIAGNOSTIC_POLICY = {
    'source': 'Read-only USD physics material binding and native tensor material storage inspection alongside legacy wrapper getters.',
    'new_diagnostic_queries_are_read_only': True,
    'effective_solver_coefficients_verified': False,
    'solver_material_consumption_verified': False,
    'interpretation': 'A resolved USD material or matching getter is not proof that PhysX is using those coefficients.',
    'backend_readback_interpretation': 'Native tensor values describe backend material storage, not effective contact coefficients or a certified solver outcome.',
}


def _vector(value, size):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'Expected {size} finite native coordinates.')
    return result


def project_release_setup(surface_world, world_aabb_center, object_position, object_quaternion,
                          ramp_position, ramp_quaternion, release_local_x_m, half_ramp_thickness_m):
    """Translate actual surface support to the release point without rotating it."""
    points = np.asarray(surface_world, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Actual native surface vertices must be a nonempty finite N by 3 array.')
    centre, position = _vector(world_aabb_center, 3), _vector(object_position, 3)
    quaternion = _vector(object_quaternion, 4)
    ramp_position, ramp_quaternion = _vector(ramp_position, 3), _vector(ramp_quaternion, 4)
    if np.linalg.norm(quaternion) == 0 or np.linalg.norm(ramp_quaternion) == 0:
        raise ValueError('Native quaternions must be nonzero.')
    if not math.isfinite(release_local_x_m) or not math.isfinite(half_ramp_thickness_m) or half_ramp_thickness_m <= 0:
        raise ValueError('Release location and positive ramp half-thickness must be measurable.')
    rotation = Rotation.from_quat(ramp_quaternion)
    local_surface = rotation.inv().apply(points - ramp_position)
    local_centre = rotation.inv().apply(centre - ramp_position)
    delta_local = np.array([release_local_x_m - local_centre[0], -local_centre[1],
                           half_ramp_thickness_m - local_surface[:, 2].min() + SETUP_CLEARANCE_M])
    delta_world = rotation.apply(delta_local)
    return {
        'position_world_m': (position + delta_world).tolist(),
        'quaternion_xyzw': quaternion.tolist(),
        'delta_ramp_local_m': delta_local.tolist(), 'delta_world_m': delta_world.tolist(),
        'world_aabb_center_before_m': centre.tolist(),
        'aabb_center_ramp_local_before_m': local_centre.tolist(),
        'predicted_aabb_center_ramp_local_m': (local_centre + delta_local).tolist(),
        'surface_minimum_ramp_local_z_before_m': float(local_surface[:, 2].min()),
        'predicted_surface_minimum_ramp_local_z_m': float(local_surface[:, 2].min() + delta_local[2]),
        'half_ramp_thickness_m': float(half_ramp_thickness_m),
        'setup_clearance_m': SETUP_CLEARANCE_M,
    }


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value))


def summarize_passive_trial(trial, *, duration_s):
    """Separate measured passive eligibility from a receiver arrival outcome."""
    reasons = list(trial.get('errors', []))
    protocol = trial.get('release_protocol', DEFAULT_RELEASE_PROTOCOL)
    if protocol not in RELEASE_PROTOCOLS:
        reasons.append('unknown_release_protocol')
    if protocol == 'contact_then_zero_velocity' and trial.get('release_protocol_evidence', {}).get('accepted') is not True:
        reasons.append('resting_release_invariants_failed_or_unavailable')
    release = trial.get('release_measurement')
    release_time = trial.get('release_native_time_s')
    if not isinstance(release, dict) or release.get('release_ready') is not True:
        reasons.append('native_release_not_ready_or_unavailable')
    if not _finite(release_time):
        reasons.append('release_native_time_unavailable')
    samples = trial.get('samples', [])
    if not isinstance(samples, list):
        samples = []
        reasons.append('passive_samples_unavailable')
    if len(samples) < MINIMUM_SAMPLES:
        reasons.append('insufficient_native_samples')
    previous = release_time if _finite(release_time) else None
    gaps, raw_arrivals = [], []
    measurements = [release] if isinstance(release, dict) else []
    for sample in samples:
        now, measured = sample.get('native_time_s'), sample.get('measurement')
        if not _finite(now):
            reasons.append('sample_native_time_unavailable')
        elif previous is not None:
            difference = now - previous
            if difference <= 0:
                reasons.append('sample_native_time_not_strictly_increasing')
            else:
                gaps.append(difference)
            previous = now
        else:
            previous = now
        if sample.get('measurement_error') or not isinstance(measured, dict):
            reasons.append('native_measurement_unavailable')
            continue
        measurements.append(measured)
        if measured.get('in_receiver') is True and _finite(now) and _finite(release_time):
            raw_arrivals.append(now - release_time)
    for measured in measurements:
        if measured.get('safe') is not True:
            reasons.append('unsafe_or_unavailable_native_sample')
        if not isinstance(measured.get('interference'), list) or measured['interference']:
            reasons.append('interference_or_unavailable_native_sample')
        if type(measured.get('in_receiver')) is not bool:
            reasons.append('receiver_measurement_unavailable')
        if any(not _finite(measured.get(key)) or measured[key] < 0
               for key in ('linear_speed_m_s', 'angular_speed_rad_s')):
            reasons.append('native_speed_measurement_unavailable')
    max_gap = max(gaps) if gaps else None
    if max_gap is None or max_gap > MAXIMUM_SAMPLE_GAP_S:
        reasons.append('native_sample_gap_exceeded_or_unavailable')
    last = samples[-1] if samples else {}
    last_time = last.get('native_time_s')
    elapsed = last_time - release_time if _finite(last_time) and _finite(release_time) else None
    if elapsed is None or elapsed < duration_s:
        reasons.append('passive_observation_duration_incomplete')
    final_measurement = last.get('measurement')
    final_receiver = final_measurement.get('in_receiver') if isinstance(final_measurement, dict) else None
    if isinstance(release, dict) and release.get('in_receiver') is True:
        raw_arrivals.append(0.)
    arrival = min(raw_arrivals) if raw_arrivals else None
    reasons = list(dict.fromkeys(reasons))
    return {
        **NON_AGENT, 'release_protocol': protocol,
        'setter_counts': deepcopy(trial.get('setter_counts')),
        'passive_calibration_eligible': not reasons,
        'eligibility_failures': reasons, 'sample_count': len(samples),
        'maximum_sample_gap_s': max_gap, 'observed_duration_s': elapsed,
        'arrival_s': arrival, 'final_in_receiver': final_receiver,
        'native_release_ready': release.get('release_ready') if isinstance(release, dict) else None,
    }


def _timing_statistics(values):
    return {'count': len(values), 'minimum_s': min(values) if values else None,
            'maximum_s': max(values) if values else None,
            'median_s': statistics.median(values) if values else None,
            'spread_s': max(values) - min(values) if values else None}


def _outcome_available(summary):
    return (isinstance(summary, dict) and 'arrival_s' in summary
        and (summary['arrival_s'] is None or (_finite(summary['arrival_s']) and summary['arrival_s'] >= 0))
        and type(summary.get('final_in_receiver')) is bool)


def aggregate_passive_trials(trials, candidates, *, repetitions, tie_tolerance_s):
    """Report complete repeated comparisons without deriving a median winner."""
    if type(repetitions) is not int or repetitions < 1 or not _finite(tie_tolerance_s) or tie_tolerance_s < 0:
        raise ValueError('Repetitions and the declared arrival tie tolerance must be valid.')
    candidates = tuple(candidates)
    protocols = {row.get('release_protocol', DEFAULT_RELEASE_PROTOCOL) for row in trials}
    if len(protocols) > 1:
        raise ValueError('Passive comparisons must use one matching release protocol.')
    groups = {(key, rep): [row for row in trials if row.get('candidate') == key and row.get('repetition') == rep]
              for key in candidates for rep in range(1, repetitions + 1)}
    comparisons, complete_sets = [], []
    for repetition in range(1, repetitions + 1):
        rows, failures = {}, {}
        for key in candidates:
            matching = groups[key, repetition]
            summary = matching[0].get('summary') if len(matching) == 1 else None
            if (not _outcome_available(summary) or summary.get('passive_calibration_eligible') is not True):
                failures[key] = ('missing_or_duplicate_trial' if len(matching) != 1 else
                                 summary.get('eligibility_failures') or ['summary_or_native_outcome_unavailable']
                                 if isinstance(summary, dict) else ['summary_unavailable'])
            rows[key] = summary
        complete = not failures
        arrivals = {key: row['arrival_s'] for key, row in rows.items() if isinstance(row, dict)
                    and row.get('passive_calibration_eligible') is True and row.get('final_in_receiver') is True
                    and _finite(row.get('arrival_s')) and row['arrival_s'] >= 0}
        admissible = None
        if complete:
            best = min(arrivals.values()) if arrivals else None
            admissible = [key for key in candidates if key in arrivals and arrivals[key] <= best + tie_tolerance_s]
            complete_sets.append(set(admissible))
        comparisons.append({'repetition': repetition, 'complete_eligible_comparison': complete,
                            'eligibility_failures': failures, 'receiver_finishing_arrival_s': arrivals,
                            'admissible_set': admissible})
    descriptive = {}
    for key in candidates:
        rows = [groups[key, rep][0]['summary'] if len(groups[key, rep]) == 1 and
                isinstance(groups[key, rep][0].get('summary'), dict) else {}
                for rep in range(1, repetitions + 1)]
        raw = [row.get('arrival_s') if _finite(row.get('arrival_s')) and row['arrival_s'] >= 0 else None
               for row in rows]
        raw_errors = ['invalid_arrival_measurement' if row.get('arrival_s') is not None and
                      (not _finite(row['arrival_s']) or row['arrival_s'] < 0) else None for row in rows]
        eligible = [row for row in rows if _outcome_available(row) and row.get('passive_calibration_eligible') is True]
        finishing = [row['arrival_s'] for row in eligible if row.get('final_in_receiver') is True
                     and _finite(row.get('arrival_s'))]
        descriptive[key] = {
            'requested_repetitions': repetitions,
            'observed_trial_count': sum(len(groups[key, rep]) == 1 for rep in range(1, repetitions + 1)),
            'eligible_trial_count': len(eligible), 'ineligible_or_missing_trial_count': repetitions - len(eligible),
            'raw_arrival_s_by_repetition': raw,
            'raw_arrival_measurement_errors_by_repetition': raw_errors,
            'raw_arrival_statistics': _timing_statistics([value for value in raw if _finite(value)]),
            'eligible_finishing_arrival_statistics': _timing_statistics(finishing),
            'eligible_no_arrival_count': sum(row.get('arrival_s') is None for row in eligible),
            'eligible_not_final_in_receiver_count': sum(row.get('final_in_receiver') is not True for row in eligible),
            'eligible_finishing_observation_count': len(finishing),
        }
    all_complete = len(complete_sets) == repetitions
    repeatable = bool(all_complete and all(value == complete_sets[0] for value in complete_sets))
    intersection = set.intersection(*complete_sets) if complete_sets else None
    union = set.union(*complete_sets) if complete_sets else None
    return {
        **NON_AGENT, 'release_protocol': next(iter(protocols), DEFAULT_RELEASE_PROTOCOL),
        'per_repetition': comparisons, 'per_candidate': descriptive,
        'all_requested_repetitions_complete': all_complete,
        'complete_repetition_count': len(complete_sets), 'repeatable': repeatable,
        'complete_repetition_admissible_intersection': [key for key in candidates if key in intersection] if intersection is not None else None,
        'complete_repetition_admissible_union': [key for key in candidates if key in union] if union is not None else None,
        'repeatable_admissible_set': comparisons[0]['admissible_set'] if repeatable else None,
        'status': ('inconclusive' if not all_complete or not union else
                   'repeatable_passive_outcome' if repeatable else 'nonrepeatable_passive_outcomes'),
        'arrival_tie_tolerance_s': tie_tolerance_s,
        'timing_statistics_scope': 'Descriptive only; medians never select an overall winner.',
        'winner_reversal': 'not_established_from_one_material_condition',
    }


def _surface_for_setup(obj, ramp_quaternion):
    """Use native collision surfaces, with an exact uniform-sphere fallback."""
    from wia.tasks.interaction_setup import array
    from wia.authoring.converters.rigid_partial_insertion import _surface_points
    try:
        points = array(_surface_points(obj))
        return points, {'method': 'native_collision_vertices_and_face_centroids',
                        'surface_sample_count': len(points)}
    except (AttributeError, ValueError, RuntimeError, TypeError) as error:
        if getattr(obj, '_primitive_type', None) != 'Sphere':
            raise
        if obj.visual_only or not any(mesh.collision_enabled for link in obj.links.values()
                                      for mesh in link.collision_meshes.values()):
            raise ValueError('Analytic sphere setup requires enabled native collision geometry.') from error
        scale = array(obj.root_link.get_world_scale())
        radius = float(obj.radius)
        if (scale.shape != (3,) or np.any(scale <= 0) or not np.all(scale == scale[0])
                or not np.isfinite(radius) or radius <= 0):
            raise ValueError('Native sphere radius and uniform world scale are required; no ellipsoid approximation.') from error
        centre = array(obj.get_position_orientation()[0])
        world_radius = radius * float(scale[0])
        axes = np.vstack((np.eye(3), -np.eye(3)))
        points = centre + Rotation.from_quat(ramp_quaternion).apply(axes * world_radius)
        return points, {'method': 'analytic_native_primitive_sphere_uniform_world_scale',
                        'surface_sample_count': len(points), 'native_radius_m': radius,
                        'native_world_scale': scale.tolist(), 'world_radius_m': world_radius,
                        'native_surface_error': f'{type(error).__name__}: {error}'}


def _usd_material_modules():
    """Use the already installed native USD modules without starting Kit."""
    from pxr import UsdPhysics, UsdShade
    try:
        from pxr import PhysxSchema
        physx_error = None
    except Exception as error:
        PhysxSchema, physx_error = None, f'{type(error).__name__}: {error}'
    return {'UsdPhysics': UsdPhysics, 'UsdShade': UsdShade, 'PhysxSchema': PhysxSchema,
            'physx_schema_error': physx_error}


def _material_attribute(reader, source, *, token=False, boolean=False):
    """Read a composed value and its authored flag without creating attributes."""
    result = {'source': source, 'available': False, 'value': None, 'authored': None, 'errors': []}
    try:
        attribute = reader()
        value = attribute.Get()
        if value is not None:
            if token:
                value = str(value)
            elif boolean:
                value = bool(value)
            else:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError('Material diagnostic value is nonfinite.')
            result.update(available=True, value=value)
        try:
            result['authored'] = bool(attribute.HasAuthoredValueOpinion())
        except Exception as error:
            result['errors'].append('authored_status_unavailable: ' + f'{type(error).__name__}: {error}')
    except Exception as error:
        result['errors'].append(f'{type(error).__name__}: {error}')
    return result


def _binding_relationship(relationship, usd_shade):
    result = {'source': 'USD physics binding relationship', 'available': False, 'errors': []}
    try:
        if relationship is None or not relationship:
            result['reason'] = 'No valid relationship was resolved.'
            return result
        result.update(available=True, path=str(relationship.GetPath()),
                      prim_path=str(relationship.GetPrim().GetPath()),
                      targets=[str(path) for path in relationship.GetTargets()])
        try:
            authored = relationship.GetMetadata('bindMaterialAs')
            result['authored_binding_strength'] = str(authored) if authored is not None else None
            result['binding_strength'] = str(usd_shade.MaterialBindingAPI.GetMaterialBindingStrength(relationship))
        except Exception as error:
            result['errors'].append('binding_strength_unavailable: ' + f'{type(error).__name__}: {error}')
    except Exception as error:
        result['errors'].append(f'{type(error).__name__}: {error}')
    return result


def _cached_wrapper_material(mesh):
    """Inspect only the existing cache; never construct/apply a PhysicsMaterial."""
    result = {'source': 'Existing mesh._applied_physics_material wrapper getters after the legacy friction read',
              'available': False, 'material_path': None, 'attributes': {}, 'errors': []}
    try:
        material = getattr(mesh, '_applied_physics_material', None)
        if material is None:
            result['reason'] = 'No existing cached physics-material wrapper is exposed.'
            return result
        result['available'] = True
        try:
            result['material_path'] = str(material.prim_path)
        except Exception as error:
            result['errors'].append('cached_material_path_unavailable: ' + f'{type(error).__name__}: {error}')
        for key in ('static_friction', 'dynamic_friction', 'restitution'):
            row = {'source': 'cached_wrapper.get_' + key, 'available': False, 'value': None, 'errors': []}
            try:
                value = getattr(material, 'get_' + key)()
                if value is not None:
                    value = float(value)
                    if not math.isfinite(value):
                        raise ValueError('Cached material value is nonfinite.')
                    row.update(available=True, value=value)
            except Exception as error:
                row['errors'].append(f'{type(error).__name__}: {error}')
            result['attributes'][key] = row
    except Exception as error:
        result['errors'].append(f'{type(error).__name__}: {error}')
    return result


def _collision_material_binding(mesh, modules):
    """Resolve the collision prim's physics binding, including inherited bindings."""
    result = {'collision_prim_path': str(mesh.prim_path),
              'cached_wrapper': _cached_wrapper_material(mesh),
              'usd_physics_binding': {'source': "UsdShade.MaterialBindingAPI.ComputeBoundMaterial(materialPurpose='physics')",
                                      'available': False, 'errors': []},
              'effective_solver_coefficients_verified': False}
    binding = result['usd_physics_binding']
    try:
        shade, physics, physx = modules['UsdShade'], modules['UsdPhysics'], modules['PhysxSchema']
        prim = mesh.prim
        api = shade.MaterialBindingAPI(prim)
        try:
            direct = api.GetDirectBinding(materialPurpose='physics')
            direct_path = str(direct.GetMaterialPath())
            direct_rel = (direct.GetBindingRel() if hasattr(direct, 'GetBindingRel')
                          else prim.GetRelationship('material:binding:physics'))
            result['direct_physics_binding'] = {'source': 'UsdShade.MaterialBindingAPI.GetDirectBinding(physics)',
                'available': bool(direct_path), 'material_path': direct_path or None,
                'relationship': _binding_relationship(direct_rel, shade)}
        except Exception as error:
            result['direct_physics_binding'] = {'source': 'UsdShade.MaterialBindingAPI.GetDirectBinding(physics)',
                'available': False, 'error': f'{type(error).__name__}: {error}'}
        resolved = api.ComputeBoundMaterial(materialPurpose='physics')
        material, relationship = resolved if isinstance(resolved, tuple) else (resolved, None)
        binding['resolved_relationship'] = _binding_relationship(relationship, shade)
        if material and material.GetPrim().IsValid():
            material_prim = material.GetPrim()
            binding.update(available=True, resolved_material_path=str(material_prim.GetPath()),
                           resolved_binding_is_on_collision_prim=(
                               binding['resolved_relationship'].get('prim_path') == str(prim.GetPath()))
                               if binding['resolved_relationship']['available'] else None,
                           material_api_applied=bool(material_prim.HasAPI(physics.MaterialAPI)))
            material_api = physics.MaterialAPI(material_prim)
            binding['attributes'] = {key: _material_attribute(lambda method=method: getattr(material_api, method)(),
                'UsdPhysics.MaterialAPI.' + method) for key, method in (
                    ('static_friction', 'GetStaticFrictionAttr'), ('dynamic_friction', 'GetDynamicFrictionAttr'),
                    ('restitution', 'GetRestitutionAttr'))}
            if physx is not None:
                try:
                    physx_api = physx.PhysxMaterialAPI(material_prim)
                    binding['physx_material_api_applied'] = bool(material_prim.HasAPI(physx.PhysxMaterialAPI))
                    binding['combine_modes'] = {key: _material_attribute(
                        lambda method=method: getattr(physx_api, method)(),
                        'PhysxSchema.PhysxMaterialAPI.' + method, token=True) for key, method in (
                            ('friction', 'GetFrictionCombineModeAttr'), ('restitution', 'GetRestitutionCombineModeAttr'))}
                except Exception as error:
                    binding['combine_modes'] = {'available': False, 'source': 'PhysxSchema.PhysxMaterialAPI',
                                                'error': f'{type(error).__name__}: {error}'}
            else:
                binding['combine_modes'] = {'available': False, 'source': 'PhysxSchema.PhysxMaterialAPI',
                                            'error': modules.get('physx_schema_error') or 'PhysxSchema unavailable.'}
        else:
            binding['reason'] = 'No valid physics material resolves on this collision prim.'
    except Exception as error:
        binding['errors'].append(f'{type(error).__name__}: {error}')
    physics = modules.get('UsdPhysics')
    result['collision_attributes'] = {
        'collision_enabled': _material_attribute(lambda: physics.CollisionAPI(mesh.prim).GetCollisionEnabledAttr(),
            'UsdPhysics.CollisionAPI.GetCollisionEnabledAttr', boolean=True),
        'approximation': _material_attribute(lambda: physics.MeshCollisionAPI(mesh.prim).GetApproximationAttr(),
            'UsdPhysics.MeshCollisionAPI.GetApproximationAttr', token=True),
    }
    cached = result['cached_wrapper']
    comparison = {'source': 'Comparison of cached-wrapper and resolved-USD readouts, not solver coefficients',
                  'material_paths_equal': None, 'attributes': {}}
    if cached.get('material_path') is not None and binding.get('resolved_material_path') is not None:
        comparison['material_paths_equal'] = cached['material_path'] == binding['resolved_material_path']
    for key in ('static_friction', 'dynamic_friction', 'restitution'):
        first, second = cached.get('attributes', {}).get(key, {}), binding.get('attributes', {}).get(key, {})
        if first.get('available') and second.get('available'):
            comparison['attributes'][key] = {'available': True, 'cached_value': first['value'],
                'resolved_usd_value': second['value'], 'exact_values_equal': first['value'] == second['value'],
                'cached_minus_resolved': first['value'] - second['value']}
        else:
            comparison['attributes'][key] = {'available': False}
    result['cached_vs_resolved'] = comparison
    return result


def _backend_material_readback(link):
    """Copy native material storage without changing bodies, materials or time.

    Only an already loaded OmniGibson simulator may supply a fallback exact-path
    view. Shape slots have no verified collision-mesh identity or active-slot
    mask here; even a successful read is not solver-consumption certification.
    """
    result = {
        'source': 'omni.physics.tensors.RigidBodyView.get_material_properties',
        'available': False, 'link_path': None,
        'link_class': type(link).__module__ + '.' + type(link).__qualname__,
        'rigid_body_enabled': _material_attribute(
            lambda: link.prim.GetAttribute('physics:rigidBodyEnabled'),
            'Link USD physics:rigidBodyEnabled', boolean=True),
        'kinematic_enabled': _material_attribute(
            lambda: link.prim.GetAttribute('physics:kinematicEnabled'),
            'Link USD physics:kinematicEnabled', boolean=True),
        'is_meta_link': None, 'collision_meshes': [],
        'view_source': None, 'view_request_path': None, 'view_count': None,
        'view_prim_paths': None, 'view_max_shapes': None, 'view_link_index': None,
        'view_attempts': [], 'channels': ['static_friction', 'dynamic_friction', 'restitution'],
        'shape_slot_mapping': {'available': False,
            'reason': 'The tensor API does not expose a verified shape-slot-to-collision-mesh mapping or active-slot mask here; max_shapes can include padding.'},
        'effective_solver_coefficients_verified': False,
        'solver_material_consumption_verified': False, 'errors': [],
    }
    try:
        path = str(link.prim_path)
        result['link_path'] = path
        if not path.startswith('/') or any(token in path for token in ('*', '?', '[', ']')):
            raise ValueError('A concrete absolute link prim path without wildcard tokens is required.')
        meta = link.is_meta_link
        if not isinstance(meta, (bool, np.bool_)):
            raise ValueError('The link meta-link flag is unavailable or not boolean.')
        result['is_meta_link'] = bool(meta)
        for mesh in link.collision_meshes.values():
            row = {'path': str(mesh.prim_path), 'collision_enabled': None, 'errors': []}
            try:
                enabled = mesh.collision_enabled
                if not isinstance(enabled, (bool, np.bool_)):
                    raise ValueError('The collision-enabled flag is unavailable or not boolean.')
                row['collision_enabled'] = bool(enabled)
            except Exception as error:
                row['errors'].append(f'{type(error).__name__}: {error}')
            result['collision_meshes'].append(row)
        if result['is_meta_link']:
            result['reason'] = 'Meta links are excluded from backend material readback.'
            return result
        if not any(row['collision_enabled'] is True for row in result['collision_meshes']):
            result['reason'] = 'No enabled collision mesh is measurable on this link.'
            return result

        view = getattr(getattr(link, '_rigid_prim_view', None), '_physics_view', None)
        if view is not None:
            attempt = {'source': 'link._rigid_prim_view._physics_view', 'valid': False}
            result['view_attempts'].append(attempt)
            try:
                attempt['valid'] = bool(view.check())
            except Exception as error:
                attempt['error'] = f'{type(error).__name__}: {error}'
            if attempt['valid']:
                result['view_source'] = attempt['source']
            else:
                view = None
        if view is None:
            result['view_source'] = 'og.sim.physics_sim_view.create_rigid_body_view(exact_link_path)'
            result['view_request_path'] = path
            attempt = {'source': result['view_source'], 'requested_path': path, 'valid': False}
            result['view_attempts'].append(attempt)
            try:
                og = sys.modules.get('omnigibson')
                if og is None:
                    raise RuntimeError('OmniGibson is not already loaded; diagnostic will not import or start it.')
                simulation_view = og.sim.physics_sim_view
                if simulation_view is None or not simulation_view.is_valid:
                    raise RuntimeError('No existing valid native simulation tensor view is available.')
                view = simulation_view.create_rigid_body_view(path)
                attempt['valid'] = bool(view.check())
                if not attempt['valid']:
                    raise RuntimeError('The exact-path native rigid-body tensor view is invalid.')
            except Exception as error:
                attempt['error'] = f'{type(error).__name__}: {error}'
                raise

        count, max_shapes = view.count, view.max_shapes
        for name, value in (('count', count), ('max_shapes', max_shapes)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f'Native view {name} must be an integer.')
        count, max_shapes = int(count), int(max_shapes)
        paths = view.prim_paths
        if isinstance(paths, str):
            raise ValueError('Native view prim_paths must be a sequence of exact prim paths.')
        paths = [str(value) for value in paths]
        result.update(view_count=count, view_max_shapes=max_shapes, view_prim_paths=paths)
        if count <= 0 or max_shapes <= 0:
            raise ValueError('The native rigid-body view has no bodies or no material shape slots; static geometry may be unsupported.')
        if len(paths) != count or len(set(paths)) != count or paths.count(path) != 1:
            raise ValueError('Native view count and unique exact prim_paths do not identify this link once.')
        if result['view_request_path'] is not None and paths != [path]:
            raise ValueError('The newly created exact-path view did not resolve only the requested link.')
        index = paths.index(path)
        result['view_link_index'] = index
        raw = view.get_material_properties()
        if hasattr(raw, 'detach'):
            raw = raw.detach().cpu().numpy()
        elif hasattr(raw, 'numpy'):
            raw = raw.numpy()
        values = np.array(raw, dtype=float, copy=True)
        result['array_shape'] = list(values.shape)
        if values.shape != (count, max_shapes, 3):
            raise ValueError('Native material array shape does not equal (count, max_shapes, 3).')
        if not np.isfinite(values).all():
            raise ValueError('Native material array contains nonfinite values.')
        result.update(available=True, copied_from_reusable_buffer=True,
                      raw_shape_slot_material_properties=values.tolist(),
                      link_shape_slot_material_properties=values[index].tolist())
    except Exception as error:
        result['errors'].append(f'{type(error).__name__}: {error}')
        if result['rigid_body_enabled'].get('value') is False:
            result['reason'] = 'Rigid-body-disabled/static link has no usable backend material readback; it was not enabled or modified.'
        else:
            result['reason'] = 'Native tensor material storage is unavailable for this link.'
    return result


def _materials(objects):
    from wia.authoring.converters.physical_dynamics_inclined_plane import _friction
    result = {}
    try:
        modules, module_error = _usd_material_modules(), None
    except Exception as error:
        modules, module_error = None, f'{type(error).__name__}: {error}'
    for key, obj in objects.items():
        try:
            result[key] = {'available': True, 'collision_materials': _friction(obj)}
        except (AttributeError, ValueError, RuntimeError, TypeError) as error:
            result[key] = {'available': False, 'measurement_error': f'{type(error).__name__}: {error}'}
        backend = {'source': 'Per-link native PhysX tensor material storage readback',
                   'available': False, 'links': [], 'errors': [],
                   'effective_solver_coefficients_verified': False,
                   'solver_material_consumption_verified': False}
        result[key]['backend_material_readback'] = backend
        try:
            backend['links'] = [_backend_material_readback(link) for link in obj.links.values()]
            backend['available'] = any(row['available'] for row in backend['links'])
        except Exception as error:
            backend['errors'].append(f'{type(error).__name__}: {error}')
        diagnostics = {**MATERIAL_BINDING_DIAGNOSTIC_POLICY, 'available': False,
                       'collision_prims': [], 'errors': []}
        result[key]['usd_binding_diagnostics'] = diagnostics
        try:
            for link in obj.links.values():
                for mesh in link.collision_meshes.values():
                    if modules is None:
                        diagnostics['collision_prims'].append({'collision_prim_path': str(mesh.prim_path),
                            'cached_wrapper': _cached_wrapper_material(mesh),
                            'usd_physics_binding': {'available': False, 'error': module_error}})
                    else:
                        diagnostics['collision_prims'].append(_collision_material_binding(mesh, modules))
            diagnostics['available'] = any(row['usd_physics_binding']['available']
                                           for row in diagnostics['collision_prims'])
            if module_error:
                diagnostics['errors'].append(module_error)
        except Exception as error:
            diagnostics['errors'].append(f'{type(error).__name__}: {error}')
    return result


def _sample_native(evaluator, candidate):
    from wia.authoring.converters.physical_dynamics_inclined_plane import _native_time
    from wia.tasks.interaction_setup import array
    sample = {'native_time_s': None, 'measurement': None}
    errors = []
    try:
        sample['native_time_s'] = _native_time()['time_s']
    except Exception as error:
        errors.append(f'{type(error).__name__}: {error}')
    try:
        sample['native_contacts'] = [
            {'body0': str(row.body0), 'body1': str(row.body1),
             'position_m': array(row.position).tolist(), 'normal': array(row.normal).tolist()}
            for row in evaluator.objects[candidate].contact_list()]
    except Exception as error:
        sample['contact_diagnostic_error'] = f'{type(error).__name__}: {error}'
    try:
        measured = evaluator._measure(candidate)
        json.dumps(measured, allow_nan=False)
        sample['measurement'] = measured
    except Exception as error:
        errors.append(f'{type(error).__name__}: {error}')
    if errors:
        sample['measurement_error'] = '; '.join(errors)
    return sample


def _hold_step(session, controller):
    """Advance one native environment/controller step without a public action."""
    session.step_control({session.robot.name: controller.hold_action.copy()})


def _counted_setter(trial, name, stage, operation):
    """Count every explicit candidate setter attempt, including partial failures."""
    counts = trial.setdefault('setter_counts', {'attempted': {}, 'completed': {}})
    for category in ('attempted', 'completed'):
        for key in ('pose', 'linear_velocity', 'angular_velocity'):
            counts[category].setdefault(key, 0)
    counts['attempted'][name] += 1
    event = {'setter': name, 'stage': stage, 'completed': False}
    trial.setdefault('setter_events', []).append(event)
    try:
        operation()
    except Exception as error:
        event['error'] = f'{type(error).__name__}: {error}'
        raise
    counts['completed'][name] += 1
    event['completed'] = True
    trial['setup_setters_completed'] = deepcopy(counts['completed'])


def _zero_candidate_velocities(obj, trial, stage, zero_like):
    _counted_setter(trial, 'linear_velocity', stage,
                   lambda: obj.set_linear_velocity(zero_like(obj.get_linear_velocity())))
    _counted_setter(trial, 'angular_velocity', stage,
                   lambda: obj.set_angular_velocity(zero_like(obj.get_angular_velocity())))


def _release_snapshot(evaluator, candidate):
    """Read pose, actual velocity vectors, contacts and the unmodified evaluator."""
    from wia.tasks.interaction_setup import array
    obj = evaluator.objects[candidate]
    snapshot = {'sample': _sample_native(evaluator, candidate)}
    try:
        position, quaternion = (array(value) for value in obj.get_position_orientation())
        snapshot.update(position_world_m=_vector(position, 3).tolist(),
                        quaternion_xyzw=_vector(quaternion, 4).tolist(),
                        linear_velocity_m_s=_vector(array(obj.get_linear_velocity()), 3).tolist(),
                        angular_velocity_rad_s=_vector(array(obj.get_angular_velocity()), 3).tolist())
    except Exception as error:
        snapshot['state_measurement_error'] = f'{type(error).__name__}: {error}'
    return snapshot


def sample_protocol_release(evaluator, candidate, trial, *, release_protocol=DEFAULT_RELEASE_PROTOCOL,
                            zero_like=None):
    """Apply only the selected passive release protocol after the contact step.

    The default is the original single sample with no additional setters/steps.
    The alternative resets candidate velocities once, never its pose, and uses
    the actual evaluator result at that same native time without overriding it.
    """
    if release_protocol not in RELEASE_PROTOCOLS:
        raise ValueError('Unknown passive release protocol.')
    trial['release_protocol'] = release_protocol
    if release_protocol == DEFAULT_RELEASE_PROTOCOL:
        return _sample_native(evaluator, candidate)
    evidence = {'protocol': release_protocol, 'accepted': False, 'checks': {},
                'pose_setters_at_final_reset': 0, 'additional_physics_steps': 0,
                'tolerances': {'position_m': RELEASE_POSITION_TOLERANCE_M,
                               'orientation_rad': RELEASE_ORIENTATION_TOLERANCE_RAD,
                               'zero_velocity': RELEASE_ZERO_VELOCITY_TOLERANCE,
                               'native_clock_s': RELEASE_CLOCK_TOLERANCE_S}}
    trial['release_protocol_evidence'] = evidence
    release = None
    try:
        evidence['before_final_velocity_reset'] = before = _release_snapshot(evaluator, candidate)
        if before.get('state_measurement_error') or before['sample'].get('measurement_error'):
            raise ValueError('Native pre-reset pose, velocity or evaluator evidence is unavailable.')
        if zero_like is None:
            import torch
            zero_like = torch.zeros_like
        _zero_candidate_velocities(evaluator.objects[candidate], trial, 'after_contact_refresh', zero_like)
        evidence['after_final_velocity_reset'] = after = _release_snapshot(evaluator, candidate)
        release = after['sample']
        if after.get('state_measurement_error') or release.get('measurement_error'):
            raise ValueError('Native post-reset pose, velocity or evaluator evidence is unavailable.')
        before_time, after_time = before['sample'].get('native_time_s'), release.get('native_time_s')
        position_delta = float(np.linalg.norm(_vector(after['position_world_m'], 3) - _vector(before['position_world_m'], 3)))
        orientation_delta = float((Rotation.from_quat(before['quaternion_xyzw']).inv() *
                                   Rotation.from_quat(after['quaternion_xyzw'])).magnitude())
        linear_speed = float(np.linalg.norm(_vector(after['linear_velocity_m_s'], 3)))
        angular_speed = float(np.linalg.norm(_vector(after['angular_velocity_rad_s'], 3)))
        measured = release.get('measurement') or {}
        evidence.update(position_change_m=position_delta, orientation_change_rad=orientation_delta,
                        readback_linear_speed_m_s=linear_speed, readback_angular_speed_rad_s=angular_speed)
        evidence['checks'] = {
            'same_native_time': _finite(before_time) and _finite(after_time)
                and abs(after_time - before_time) <= RELEASE_CLOCK_TOLERANCE_S,
            'position_unchanged': position_delta <= RELEASE_POSITION_TOLERANCE_M,
            'orientation_unchanged': orientation_delta <= RELEASE_ORIENTATION_TOLERANCE_RAD,
            'actual_linear_velocity_zero': linear_speed <= RELEASE_ZERO_VELOCITY_TOLERANCE,
            'actual_angular_velocity_zero': angular_speed <= RELEASE_ZERO_VELOCITY_TOLERANCE,
            'evaluator_velocities_zero': all(_finite(measured.get(key)) and
                0 <= measured[key] <= RELEASE_ZERO_VELOCITY_TOLERANCE
                for key in ('linear_speed_m_s', 'angular_speed_rad_s')),
            'native_contact_diagnostics_available': all(isinstance(value['sample'].get('native_contacts'), list)
                and not value['sample'].get('contact_diagnostic_error') for value in (before, after)),
        }
        evidence['accepted'] = all(evidence['checks'].values())
        if not evidence['accepted']:
            trial['errors'].append('resting_release_readback_invariants_failed')
    except Exception as error:
        evidence['error'] = f'{type(error).__name__}: {error}'
        trial['errors'].append('resting_release_invariants_unavailable: ' + evidence['error'])
    return release if release is not None else _sample_native(evaluator, candidate)


def _run_trial(session, controller, evaluator, baseline, candidate, repetition, directory, result,
               *, release_protocol=DEFAULT_RELEASE_PROTOCOL):
    import torch
    from wia.runtime.scene_capture_queue import write_json
    from wia.tasks.interaction_setup import array
    from wia.authoring.converters.physical_dynamics_inclined_plane import _native_time, _pose

    directory.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    trial = {**NON_AGENT, 'candidate': candidate, 'repetition': repetition, 'attempted': True,
             'release_protocol': release_protocol,
             'samples': [], 'errors': [],
             'setup_kind': 'explicit_environment_pose_and_zero_velocity_setters',
             'native_measurement_function': 'InclinedComparisonEvaluator._measure',
             'original_baseline_quaternion_xyzw': evaluator.initial[candidate]['quaternion_xyzw']}
    result['trials'].append(trial)
    sim, obj, ramp = session.pipeline.og.sim, evaluator.objects[candidate], evaluator.objects['ramp']
    try:
        before_restore = _native_time()
        sim.load_state(deepcopy(baseline), serialized=False)
        result['baseline_restore_count'] += 1
        trial['baseline_restore_index'] = result['baseline_restore_count']
        _hold_step(session, controller)
        trial['baseline_restore'] = {'complete': True, 'serialized': False,
            'refresh_steps': 1, 'clock_before_restore': before_restore, 'clock_after_refresh': _native_time(),
            'reason': 'Installed OG requires one native step after load_state to refresh contact-dependent states.'}
    except Exception as error:
        trial['errors'].append('baseline_restore_or_refresh_failed: ' + f'{type(error).__name__}: {error}')
        trial['summary'] = summarize_passive_trial(trial, duration_s=evaluator.p['trial_duration_s'])
        write_json(directory / 'trial.json', trial)
        raise RuntimeError('Full native baseline restoration failed; remaining passive trials were not attempted.') from error
    trial['materials_before_setup'] = _materials({key: evaluator.objects[key] for key in ('ramp', candidate)})
    try:
        before = _pose(obj)
        json.dumps(before, allow_nan=False)
        trial['before_setup'] = before
        ramp_position, ramp_quaternion = (array(value) for value in ramp.get_position_orientation())
        points, method = _surface_for_setup(obj, ramp_quaternion)
        centre = (array(before['aabb_min']) + array(before['aabb_max'])) / 2
        plan = project_release_setup(points, centre, before['position'], before['quaternion_xyzw'],
            ramp_position, ramp_quaternion, evaluator.p['release_local_x_m'], evaluator.p['ramp_dimensions_m'][2] / 2)
        trial['setup_projection'], trial['surface_evidence'] = plan, method
        trial['post_refresh_resting_quaternion_xyzw'] = before['quaternion_xyzw']
        trial['setup_native_time_s'] = _native_time()['time_s']
        native_position, native_quaternion = obj.get_position_orientation()
        _counted_setter(trial, 'pose', 'projection', lambda: obj.set_position_orientation(
            position=torch.as_tensor(plan['position_world_m'], dtype=native_position.dtype, device=native_position.device),
            orientation=torch.as_tensor(plan['quaternion_xyzw'], dtype=native_quaternion.dtype, device=native_quaternion.device)))
        _zero_candidate_velocities(obj, trial, 'projection', torch.zeros_like)
        after = _pose(obj)
        json.dumps(after, allow_nan=False)
        trial['after_setup_before_contact_step'] = after
        actual_centre = (array(after['aabb_min']) + array(after['aabb_max'])) / 2
        actual_local = Rotation.from_quat(ramp_quaternion).inv().apply(actual_centre - ramp_position)
        before_rotation = Rotation.from_quat(before['quaternion_xyzw'])
        after_rotation = Rotation.from_quat(after['quaternion_xyzw'])
        trial['actual_setup_deviation_before_contact_step'] = {
            'aabb_center_ramp_local_m': actual_local.tolist(),
            'center_x_error_m': float(actual_local[0] - evaluator.p['release_local_x_m']),
            'center_y_error_m': float(actual_local[1]),
            'orientation_change_from_post_refresh_resting_rad': float((before_rotation.inv() * after_rotation).magnitude()),
        }
    except Exception as error:
        trial['errors'].append('setup_failed: ' + f'{type(error).__name__}: {error}')
    try:
        _hold_step(session, controller)
        trial['setup_contact_refresh_steps'] = 1
    except Exception as error:
        trial['errors'].append('contact_refresh_failed: ' + f'{type(error).__name__}: {error}')
    release = sample_protocol_release(evaluator, candidate, trial, release_protocol=release_protocol)
    trial['release_native_time_s'], trial['release_measurement'] = release['native_time_s'], release['measurement']
    trial['release_contact_diagnostics'] = {key: release[key] for key in
        ('native_contacts', 'contact_diagnostic_error') if key in release}
    if release.get('measurement_error'):
        trial['errors'].append('release_measurement_failed: ' + release['measurement_error'])
    if release['measurement'] is not None:
        measured = release['measurement']
        q = Rotation.from_quat(measured['quaternion_xyzw'])
        reference = Rotation.from_quat(trial['original_baseline_quaternion_xyzw'])
        resting = Rotation.from_quat(trial.get('post_refresh_resting_quaternion_xyzw', trial['original_baseline_quaternion_xyzw']))
        local = measured['local_ramp_position']
        trial['actual_release_deviation'] = {
            'ramp_local_center_x_error_m': local[0] - evaluator.p['release_local_x_m'],
            'ramp_local_center_y_error_m': local[1],
            'orientation_from_original_baseline_rad': float((reference.inv() * q).magnitude()),
            'orientation_from_post_refresh_resting_rad': float((resting.inv() * q).magnitude()),
            'native_release_ready': measured['release_ready'],
        }
    duration = evaluator.p['trial_duration_s']
    limit = math.ceil(duration / float(sim.get_sim_step_dt())) + 1
    with (directory / 'trace.jsonl').open('x') as stream:
        for index in range(limit):
            try:
                _hold_step(session, controller)
                sample = _sample_native(evaluator, candidate)
            except Exception as error:
                sample = {'native_time_s': None, 'measurement': None,
                          'measurement_error': f'{type(error).__name__}: {error}'}
            sample['controller_step'] = index + 1
            start, now = trial['release_native_time_s'], sample['native_time_s']
            sample['elapsed_s'] = now - start if _finite(now) and _finite(start) else None
            trial['samples'].append(sample)
            stream.write(json.dumps(sample, allow_nan=False) + '\n')
            stream.flush()
            if sample['elapsed_s'] is not None and sample['elapsed_s'] >= duration:
                break
    trial['materials_after_passive_trace'] = _materials({key: evaluator.objects[key] for key in ('ramp', candidate)})
    trial['summary'] = summarize_passive_trial(trial, duration_s=duration)
    trial['elapsed_wall_s'] = time.monotonic() - began
    write_json(directory / 'trial.json', trial)
    return trial


def _fine_controller(session):
    """Mirror the diagnosed fine controller; no trunk actions or trunk override."""
    from wia.runtime.sim.esi_base import BaseControlConfig
    from wia.runtime.sim.esi_arm import ArmControlConfig
    from wia.runtime.sim.esi_gripper import R1ManipulationController, GripperControlConfig
    from wia.runtime.sim.esi_rotation import RotationControlConfig
    from wia.runtime.sim.esi_trunk import R1TrunkController, TrunkControlConfig

    profile = json.loads((ROOT / 'configs/esi_primitives_fine.json').read_text())
    base = replace(BaseControlConfig.load(), **profile['base'])
    arm = replace(ArmControlConfig.load(), **profile['arm'])
    rotation = replace(RotationControlConfig.load(), **profile['rotation'])
    gripper = GripperControlConfig.load()
    trunk = TrunkControlConfig.load(ROOT / 'configs/esi_trunk.json')
    controller = R1ManipulationController(session, base, arm_config=arm,
        gripper_config=gripper, rotation_config=rotation)
    controller.trunk_controller = R1TrunkController(controller, trunk)
    effective = {'primitive_profile': 'fine', 'base': asdict(base), 'arm': asdict(arm),
                 'rotation': asdict(rotation), 'gripper': asdict(gripper), 'trunk': asdict(trunk),
                 'trunk_profile_policy': 'Unchanged default trunk; no trunk actions in passive calibration.',
                 'hold_action': controller.hold_action.tolist(), 'controller_dt_s': controller.dt}
    return controller, effective


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-file', required=True, type=Path)
    parser.add_argument('--scene-file', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--variant', help='Exact shared declared variant; otherwise adapted, then original.')
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--release-protocol', choices=RELEASE_PROTOCOLS, default=DEFAULT_RELEASE_PROTOCOL)
    args = parser.parse_args(argv)
    if args.repetitions < 1:
        parser.error('--repetitions must be positive.')
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    from wia.runtime.scene_capture_queue import write_json
    result = {**NON_AGENT, 'schema': SCHEMA, 'complete': False, 'task_success': False,
              'material_binding_diagnostic_metadata': deepcopy(MATERIAL_BINDING_DIAGNOSTIC_POLICY),
              'status': 'initializing_passive_calibration', 'pid': os.getpid(),
              'release_protocol': args.release_protocol,
              'repetitions': args.repetitions, 'native_reset_count': 0, 'baseline_restore_count': 0,
              'trials': [], 'errors': [], 'task_file': str(args.task_file.resolve()),
              'scene_file': str(args.scene_file.resolve())}
    began, session, candidates, parameters = time.monotonic(), None, ('trial_1', 'trial_2', 'trial_3'), None
    try:
        from wia.tasks.loader import LoadedScene, LoadedTask, UniqueLoader, sha256, task_definition, source_scene_room
        from wia.tasks.schema import SceneDefinition
        from wia.tasks.sources import resolve_source
        from wia.tasks.scene import ScenePatches
        import yaml
        from wia.runtime.commands.check_esi_core_native import select_native_variant

        task_path, scene_path = args.task_file.resolve(), args.scene_file.resolve()
        definition = SceneDefinition.model_validate(yaml.load(scene_path.read_text(), Loader=UniqueLoader))
        task_def = task_definition(yaml.load(task_path.read_text(), Loader=UniqueLoader))
        variant = select_native_variant(definition, task_def, args.variant)
        if task_def.source.json_path != definition.source.json_path:
            raise ValueError('The exact task and scene must declare the same source identity.')
        evaluation = task_def.private.evaluation
        if (evaluation.module, evaluation.operation) != (
                'wia.authoring.converters.inclined_comparison', 'controlled_trials_then_spare_delivery'):
            raise ValueError('Passive ramp calibration requires the existing controlled comparison evaluator.')
        parameters = evaluation.parameters
        if (set(pair['trial'] for pair in parameters['candidate_pairs']) != set(candidates)
                or len(parameters['candidate_pairs']) != 3 or parameters['trial_duration_s'] != 3.):
            raise ValueError('The declared three candidates and three-second protocol must remain unchanged.')
        source = resolve_source(definition.source.json_path)
        raw = json.loads(source.read_text())
        scene_name, room_name = source_scene_room(definition.source.json_path, raw)
        result.update(task_sha256=sha256(task_path), scene_sha256=sha256(scene_path), source_sha256=sha256(source),
                      source_file=str(source), source_identity=definition.source.json_path, variant=variant,
                      seed=definition.seed, parameters=deepcopy(parameters), candidates=list(candidates))
        selected = [value for value in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if value.strip()]
        if len(selected) != 1 or not os.environ.get('OMNIGIBSON_APPDATA_PATH'):
            raise ValueError('Select exactly one local CUDA GPU and isolated OMNIGIBSON_APPDATA_PATH before launch.')
        from wia.runtime.commands.run_esi_capture_worker import allocation_gpu
        from wia.runtime.scene_capture_adapters import install_gpu_selection, isolate_texture_cache, gpu_evidence
        gpu = allocation_gpu(0, execution_mode='local', renderer_order='pci')
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu['uuid']
        result['gpu'] = gpu
        install_gpu_selection(gpu)
        from omnigibson import simulator
        isolate_texture_cache(simulator, os.environ['OMNIGIBSON_APPDATA_PATH'])
        from wia.runtime.sim.esi_omnigibson import ESISession
        from wia.runtime.sim.esi_robot_cameras import configure_robot_cameras
        from wia.authoring.converters.inclined_comparison import InclinedComparisonEvaluator, qualify_operation
        from wia.authoring.converters.physical_dynamics_inclined_plane import _pose, _native_time

        loaded = LoadedScene(definition, scene_path, source, sha256(scene_path), sha256(source),
            variant, definition.seed, scene_name, room_name, run_mode='diagnostic')
        task = LoadedTask(task_def, task_path, source, sha256(task_path), sha256(source),
            variant, definition.seed, scene_name, room_name, run_mode='diagnostic')
        adapter = ScenePatches(loaded, definition.bindings)
        settings = adapter.observation_settings()
        settings['camera_size'] = list(definition.runtime.camera_size)
        session = ESISession(source, require_robot=True, grasp_mode='assisted', **settings)
        session.scene_adapter, session.loaded_task, session.public_task_text = adapter, task, task.task_text
        write_json(output / 'result.json', result)
        session.reset()
        result['native_reset_count'] += 1
        result['gpu'] = gpu_evidence(session, gpu)
        configure_robot_cameras(session, ROOT / definition.runtime.robot_camera_config,
            resolution=definition.runtime.camera_size)
        controller, effective = _fine_controller(session)
        result['effective_controller'] = effective
        config = evaluation.model_dump()
        config.update(scene=session.scene_name, room=session.room_name,
                      bindings={key: binding.name for key, binding in task_def.private.bindings.items()})
        evaluator = InclinedComparisonEvaluator(session, controller, config, output / 'measurement_only')
        qualification = qualify_operation(session, controller, config, evaluator.objects)
        result['initial_qualification'] = qualification
        result['initialization'] = {'interaction_setup': getattr(session, 'category_interaction_setup', None),
                                   'controller': effective, 'native_clock': _native_time()}
        write_json(output / 'initialization.json', result['initialization'])
        if qualification.get('accepted') is not True:
            raise ValueError('Initial native inclined-comparison layout qualification was not accepted.')
        evaluator.initial = {key: _pose(obj) for key, obj in evaluator.objects.items()}
        json.dumps(evaluator.initial, allow_nan=False)
        result['baseline_poses'] = deepcopy(evaluator.initial)
        result['actual_materials'] = _materials(evaluator.objects)
        baseline = deepcopy(session.pipeline.og.sim.dump_state(serialized=False))
        result['baseline'] = {'serialized': False, 'stored_in_memory_only': True,
                              'native_clock': _native_time(), 'full_state_restored_before_every_trial': True}
        result['measurement_evaluator_active'] = evaluator.active
        result['measurement_evaluator_begin_or_record_called'] = False
        for repetition in range(1, args.repetitions + 1):
            for candidate in candidates:
                task.verify_unchanged()
                loaded.verify_unchanged()
                print(f'Passive calibration repetition {repetition}/{args.repetitions}: {candidate}', flush=True)
                _run_trial(session, controller, evaluator, baseline, candidate, repetition,
                    output / f'repetition_{repetition:03d}' / candidate, result,
                    release_protocol=args.release_protocol)
                write_json(output / 'result.json', result)
        result['complete'] = True
    except Exception as error:
        result['errors'].append(f'{type(error).__name__}: {error}')
        result['traceback'] = traceback.format_exc()
        print(result['traceback'], flush=True)
    finally:
        if parameters is not None:
            for trial in result['trials']:
                if not isinstance(trial.get('summary'), dict):
                    trial['errors'].append('passive_trial_interrupted_before_summary')
                    trial['summary'] = summarize_passive_trial(trial, duration_s=parameters['trial_duration_s'])
            for repetition in range(1, args.repetitions + 1):
                for candidate in candidates:
                    if any(row['candidate'] == candidate and row['repetition'] == repetition for row in result['trials']):
                        continue
                    missing = {**NON_AGENT, 'candidate': candidate, 'repetition': repetition,
                               'release_protocol': args.release_protocol,
                               'attempted': False, 'samples': [],
                               'errors': ['not_attempted_due_initialization_or_execution_failure']}
                    missing['summary'] = summarize_passive_trial(missing, duration_s=parameters['trial_duration_s'])
                    result['trials'].append(missing)
            result['comparison'] = aggregate_passive_trials(result['trials'], candidates,
                repetitions=args.repetitions, tie_tolerance_s=parameters['arrival_tie_tolerance_s'])
            result['status'] = result['comparison']['status'] if result['complete'] else 'inconclusive_initialization_or_execution_failure'
        else:
            result['status'] = 'inconclusive_initialization_failure'
        result['elapsed_wall_s'] = time.monotonic() - began
        result['setter_counts'] = {category: {key: sum(row.get('setter_counts', {}).get(category, {}).get(key, 0)
            for row in result['trials']) for key in ('pose', 'linear_velocity', 'angular_velocity')}
            for category in ('attempted', 'completed')}
        result['setter_count_scope'] = ('Explicit candidate setters in this script only; native initialization '
            'and full-state restore operations are recorded separately, not expanded into implied setter counts.')
        write_json(output / 'result.json', result)
        if session is not None:
            session.close()
    return 0 if result['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
