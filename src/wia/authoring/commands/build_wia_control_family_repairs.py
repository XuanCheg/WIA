"""Audit explicit control families and build isolated, unqualified geometry variants.

Run: python -B scripts/build_wia_control_family_repairs.py [--output-dir DIR] [--check]
The default v2 manifest versions D1 free-attempt/DONE-only semantics and references
unchanged v1 D2 definitions. Pass --revision v1 for the historical geometry build.
No simulator or GPU is used. Original manifests and definitions are read-only.


Canonical invocation: python -m wia.authoring.commands.build_wia_control_family_repairs --help
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from wia.resources import ROOT
from wia.authoring.commands._inputs import ARTIFACT_ROOT
from wia.tasks.loader import UniqueLoader, load_scene, load_task
from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition

BASE = ROOT / 'explore_to_achieve/wia_v1'
DEFAULT_OUTPUT = BASE / 'control_family_repairs_v1'
DEFAULT_V2_OUTPUT = BASE / 'control_family_repairs_v2'
DIMENSIONAL_MANIFESTS = (BASE / 'dimensional_v2/manifest.json',
    BASE / 'workstation_transfer_v1/portfolio_manifest_v2.json')
TRAY_MANIFESTS = (BASE / 'furnished_v1/collection_robot_v3/manifest.json',
    BASE / 'review_20261025/history_collection_v3/manifest.json',
    BASE / 'review_20261025/t07_native_velocity_v2/manifest.json')
DIMENSIONAL_ALIASES = {'D1-1', 'D1-2', 'D1-3', 'D2-1', 'D2-2'}
TRAY_ALIASES = {'T07', 'T08', 'T35', 'T36', 'T37', 'T38', 'T39'}
VARIANTS = ('g1_closer', 'g2_front_crop', 'g3_narrow')
INITIALIZATION_VARIANT = 'i0_open'
TRAY_P1_ALIASES = {'T07', 'T08', 'T35', 'T36', 'T37'}
TRAY_P1_POSE = [.45, -.25, .85]
TRAY_P1_PROBE = ARTIFACT_ROOT / 'shared_control_runtime_20261026/family/tray_p1/probe/result.json'
FLAGS = {'native_validation_run': False, 'native_geometry_verified': False,
    'native_loaded': False, 'physical_success_verified': False,
    'scientific_controls_verified': False, 'continuous_path_certified': False}
MARGIN = .05
D1_COMMITMENT_CONSTRAINT = ('The first candidate block you grasp commits your selection. '
    'Keep the unselected block within 2 cm of its starting position.')
D1_FREE_ATTEMPT_CONSTRAINTS = (
    'Free attempts and switching between candidate blocks are allowed; the first grasp does not commit your selection.',
    'Final task evaluation occurs only when you declare DONE. The original grasp, lift, physical passage through '
    'the opening, supported blue-pad placement, release and stability success conditions still apply.',
)
D1_COMPLETION_MEANING = {
    'choice': 'Evaluated at DONE for the terminal candidate; the first grasp does not commit selection.',
    'grasp_lift': 'Evaluated at DONE from real grasp and lift evidence for the terminal candidate, using unchanged thresholds.',
    'passage': 'Evaluated at DONE from actual near-to-far aperture traversal evidence for the terminal candidate.',
    'stable_placement': 'Evaluated at DONE from supported blue-pad placement, released open/empty hands, no robot contact and unchanged stability thresholds.',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_rows(paths, aliases):
    rows = []
    for path in paths:
        value = json.loads(path.read_text())
        for row in value['entries']:
            alias = row.get('proposal_alias', row.get('document_alias'))
            if alias not in aliases:
                continue
            result = {'document_alias': alias, 'manifest_file': str(path),
                'manifest_sha256': digest(path), 'input_entry': deepcopy(row)}
            for kind in ('task', 'scene'):
                file = Path(row[kind + '_file'])
                result[kind + '_file'] = str(file.resolve() if file.is_absolute() else (path.parent / file).resolve())
            rows.append(result)
    return rows


def read_pair(row):
    task, scene = (yaml.load(Path(row[k + '_file']).read_text(), Loader=UniqueLoader)
                   for k in ('task', 'scene'))
    CategoryTaskDefinition.model_validate(task)
    SceneDefinition.model_validate(scene)
    if (task['scene'] != scene['scene'] or task['private']['bindings'] != scene['bindings']
            or task['defaults']['seed'] != scene['seed'] or task['source'] != scene['source']):
        raise ValueError('Task/scene initialization, bindings, seed or source copies differ: ' + row['task_file'])
    loaded_task = load_task(row['task_file'], variant='adapted', run_mode='diagnostic')
    loaded_scene = load_scene(row['scene_file'], variant='adapted', run_mode='diagnostic')
    if loaded_task.scene_name != loaded_scene.scene_name or loaded_task.room_name != loaded_scene.room_name:
        raise ValueError('Parent task/scene loaders disagree about source identity.')
    loaded_task.verify_unchanged()
    loaded_scene.verify_unchanged()
    return task, scene


def layouts(task):
    return [p['parameters']['layout'] for p in task['scene']['variants']['adapted']['patches']
            if 'layout' in p.get('parameters', {})]


def dimensional_layout(task):
    patches = task['scene']['variants']['adapted']['patches']
    if (len(patches) != 2 or {p['phase'] for p in patches} != {'before_construction', 'after_postprocessing'}
            or any(p['module'] != 'core.esi_converters.active_layout' for p in patches)
            or patches[0]['parameters'] != patches[1]['parameters']):
        raise ValueError('Dimensional initialization copies must match exactly.')
    result = layouts(task)[0]
    parameters = task['private']['evaluation']['parameters']
    if 'layout' in parameters and parameters['layout'] != result:
        raise ValueError('Evaluation layout differs from initialization.')
    objects = {o['name']: o for o in result['objects']}
    for key, dimensions in parameters['candidate_dimensions_m'].items():
        name = task['private']['bindings'][key]['name']
        if not np.allclose(cube_dimensions(objects[name]), dimensions, atol=1e-12, rtol=0):
            raise ValueError('Candidate dimensions disagree with evaluation.')
    return result


def cube_dimensions(obj):
    if obj.get('type') != 'PrimitiveObject' or obj.get('primitive_type') != 'Cube':
        raise ValueError('Exact footprint requires an authored Cube primitive.')
    return np.asarray(obj['scale'], dtype=float) * obj.get('size', 1.)


def footprint(obj, center=None, rotation=None):
    """Return exact oriented box footprint in the supplied table frame."""
    center = np.zeros(3) if center is None else np.asarray(center)
    rotation = np.eye(3) if rotation is None else np.asarray(rotation)
    local_center = rotation.T @ (np.asarray(obj['position']) - center)
    relative_rotation = rotation.T @ Rotation.from_quat(obj['orientation']).as_matrix()
    half = np.abs(relative_rotation) @ (cube_dimensions(obj) / 2.)
    return local_center - half, local_center + half


def geometry_audit(task, *, enforce=False):
    layout = dimensional_layout(task)
    objects = {o['name']: o for o in layout['objects']}
    table = objects[task['private']['bindings']['table']['name']]
    rotation = Rotation.from_quat(table['orientation']).as_matrix()
    approach = Rotation.from_quat(layout['robot_start']['orientation']).apply([1., 0., 0.])
    if not np.allclose(rotation, np.eye(3), atol=1e-10) or not np.allclose(approach, [1, 0, 0], atol=1e-10):
        raise ValueError('Repair requires the reviewed axis-aligned table and +X approach.')
    center, dimensions = np.asarray(table['position']), cube_dimensions(table)
    robot = rotation.T @ (np.asarray(layout['robot_start']['position']) - center)
    if robot[0] >= -dimensions[0] / 2:
        raise ValueError('Robot must start before the table front.')
    records, legs = [], []
    for obj in layout['objects']:
        if obj is table:
            continue
        low, high = footprint(obj, center, rotation)
        margins = [low[0] + dimensions[0] / 2, dimensions[0] / 2 - high[0],
                   low[1] + dimensions[1] / 2, dimensions[1] / 2 - high[1]]
        record = {'name': obj['name'], 'dimensions_m': cube_dimensions(obj).tolist(),
            'orientation_xyzw': obj['orientation'], 'bounds_table_frame_m': [low.tolist(), high.tolist()],
            'edge_margins_front_far_left_right_m': margins, 'minimum_edge_margin_m': min(margins)}
        if '_leg_' in obj['name']:
            record['under_table'] = bool(min(margins) >= -1e-10 and high[2] <= -dimensions[2] / 2 + 1e-9)
            legs.append(record)
        else:
            records.append(record)
    minimum = min(r['minimum_edge_margin_m'] for r in records)
    if enforce and (minimum < MARGIN - 1e-9 or not all(r['under_table'] for r in legs)):
        raise ValueError('Variant fails support margin or leg containment.')
    pads = [r for r in records if '_pad_' in r['name']]
    max_y = max(max(abs(r['bounds_table_frame_m'][0][1]), abs(r['bounds_table_frame_m'][1][1])) for r in records)
    return {'scene_model': layout['scene_model'], 'rooms': layout['rooms'],
        'table_center_m': center.tolist(), 'table_dimensions_m': dimensions.tolist(),
        'table_axes_world': rotation.tolist(), 'robot_start': deepcopy(layout['robot_start']),
        'robot_approach_world': approach.tolist(), 'objects': records, 'legs': legs,
        'minimum_support_margin_m': minimum, 'required_centered_width_with_margin_m': 2 * (max_y + MARGIN),
        'far_pads': pads, 'scientific_geometry_native_verified': False}


def repair_pair(parent_task, parent_scene, variant, provenance):
    if variant not in (*VARIANTS, INITIALIZATION_VARIANT):
        raise ValueError('Choose exactly one reviewed geometry variant; combinations are forbidden.')
    task, scene = deepcopy(parent_task), deepcopy(parent_scene)
    if variant != INITIALIZATION_VARIANT and task['private']['evaluation']['operation'] != 'select_pass_aperture':
        raise ValueError('Only the reviewed D1 aperture family may generate repair candidates.')
    original_layout = dimensional_layout(parent_task)
    geometry_audit(parent_task)
    layout = deepcopy(original_layout)
    objects = {o['name']: o for o in layout['objects']}
    bindings = task['private']['bindings']
    table = objects[bindings['table']['name']]
    table_center = np.asarray(table['position'])
    if variant == INITIALIZATION_VARIANT:
        pass
    elif variant == 'g1_closer':
        for key in task['private']['evaluation']['parameters']['candidates']:
            objects[bindings[key]['name']]['position'][0] -= .10
    elif variant == 'g2_front_crop':
        table['scale'][0] -= .20 / table.get('size', 1.)
        table['position'][0] += .10
        for key, binding in bindings.items():
            if key.startswith('leg_') and objects[binding['name']]['position'][0] < table_center[0]:
                objects[binding['name']]['position'][0] += .20
    else:
        if not np.isclose(cube_dimensions(table)[1], 2.40):
            raise ValueError('Reviewed narrow candidate requires the original 2.40 m width.')
        table['scale'][1] = 1.20 / table.get('size', 1.)
        for key, binding in bindings.items():
            if key.startswith('leg_'):
                leg = objects[binding['name']]
                leg['position'][1] -= float(np.sign(leg['position'][1] - table_center[1])) * .60
    suffix = ('__control_family_v1__' + layout['scene_model'] + '__seed'
              + str(task['defaults']['seed']) + '__' + variant)
    layout['shared_layout_id'] = layout.get('shared_layout_id', 'wia_workstation') + suffix
    layout['initial_gripper_state'] = 'open'
    for patch in task['scene']['variants']['adapted']['patches']:
        patch['parameters']['layout'] = deepcopy(layout)
    task['private']['evaluation']['parameters']['layout'] = deepcopy(layout)
    task.update(id=parent_task['id'] + suffix, version=parent_task['version'] + 1, status='pending')
    task['repair'].update(deepcopy(provenance), control_family_geometry_variant=variant,
        physical_repair_candidate=True, full_reference_execution='pending', **FLAGS)
    for key, value in task['repair'].items():
        if key.startswith('native_') and isinstance(value, bool):
            task['repair'][key] = False
    scene.update(id=parent_scene['id'] + suffix, version=parent_scene['version'] + 1,
        status='pending', scene=deepcopy(task['scene']))
    CategoryTaskDefinition.model_validate(task)
    SceneDefinition.model_validate(scene)
    geometry_audit(task, enforce=True)
    return task, scene


def tray_audit(row, task):
    parameters = task['private']['evaluation']['parameters']
    hooks = task['scene']['variants']['adapted']['patches']
    if 'plate' in parameters:
        selected = [p for p in hooks if 'plate' in p.get('parameters', {})]
        if len(selected) != 2 or any(p['parameters'] != parameters for p in selected):
            raise ValueError('Tray setup and evaluation copies differ.')
        if parameters['layout']['plate'] != parameters['plate']:
            raise ValueError('Tray layout/evaluation plate copies differ.')
        plate = deepcopy(parameters['plate'])
        return {**row, 'source_identity': task['source']['json'],
            'evaluation_module': task['private']['evaluation']['module'],
            'initializer': 'core.sim.esi_collection_setup.initialize_collection_tray',
            'initializer_hooks': [{'module': p['module'], 'phase': p['phase']} for p in selected],
            'plate': plate, 'robot_start': parameters['layout']['robot_start'],
            'limits': {k: v for k, v in parameters.items() if k.startswith('maximum_') or k == 'terminal_dwell_s'},
            'layout_copies_consistent': True, 'tray_initializer_present': True,
            'lineage': deepcopy(task['repair']), **FLAGS}
    # Temporal retrieval definitions use a fixed destination, not the room-collection initializer.
    copies = layouts(task)
    normalized = [{k: v for k, v in value.items() if k != 'human_review'} for value in copies]
    if len(normalized) != 2 or normalized[0] != normalized[1]:
        raise ValueError('Temporal apparatus layout copies differ beyond human review metadata.')
    objects = {o['name']: o for o in normalized[0]['objects']}
    key = parameters['destination_key']
    destination = task['private']['bindings'][key]['name']
    return {**row, 'source_identity': task['source']['json'],
        'evaluation_module': task['private']['evaluation']['module'], 'initializer': None,
        'tray_initializer_present': False, 'plate': None, 'destination_key': key,
        'destination_object': objects[destination], 'robot_start': normalized[0]['robot_start'],
        'apparatus': parameters['apparatus'], 'layout_copies_consistent_except_human_review': True,
        'lineage': deepcopy(task['repair']), **FLAGS}


def build_package(output_dir=DEFAULT_OUTPUT):
    output_dir = Path(output_dir).resolve()
    dimensional = manifest_rows(DIMENSIONAL_MANIFESTS, DIMENSIONAL_ALIASES)
    trays = manifest_rows(TRAY_MANIFESTS, TRAY_ALIASES)
    trays = [r for r in trays if r['document_alias'] != 'T07' or r['manifest_file'] == str(TRAY_MANIFESTS[-1])]
    rendered, audit_rows, tray_rows, entries = {}, [], [], []
    snapshots = {str(p): digest(p) for p in (*DIMENSIONAL_MANIFESTS, *TRAY_MANIFESTS)}
    for row in dimensional:
        task, scene = read_pair(row)
        for key in ('task', 'scene'):
            row['parent_' + key + '_sha256'] = digest(row[key + '_file'])
            snapshots[row[key + '_file']] = row['parent_' + key + '_sha256']
        audit_rows.append({**row, 'seed': task['defaults']['seed'],
            'source_identity': task['source']['json'],
            'evaluation_module': task['private']['evaluation']['module'],
            'operation': task['private']['evaluation']['operation'], 'geometry': geometry_audit(task)})
        variants = (INITIALIZATION_VARIANT, *VARIANTS) if row['document_alias'].startswith('D1-') else (INITIALIZATION_VARIANT,)
        for variant in variants:
            provenance = {'parent_task_file': row['task_file'], 'parent_scene_file': row['scene_file'],
                'parent_task_sha256': row['parent_task_sha256'], 'parent_scene_sha256': row['parent_scene_sha256'],
                'parent_manifest_file': row['manifest_file'], 'parent_manifest_sha256': row['manifest_sha256']}
            new_task, new_scene = repair_pair(task, scene, variant, provenance)
            stem = Path(row['task_file']).stem + '__' + variant
            files = {}
            for kind, definition in (('task', new_task), ('scene', new_scene)):
                path = output_dir / kind / (stem + '.yaml')
                if path in rendered:
                    raise ValueError('Duplicate output identity: ' + str(path))
                rendered[path] = yaml.safe_dump(definition, sort_keys=False, allow_unicode=False, width=88).encode()
                files[kind + '_file'] = str(path)
            entries.append({'document_alias': row['document_alias'], 'variant_id': stem,
                'geometry_variant': variant, 'seed': task['defaults']['seed'], 'variant': 'adapted',
                'task_id': new_task['id'], 'task_version': new_task['version'], 'scene_id': new_scene['id'],
                'module': new_task['private']['evaluation']['module'],
                'revision': new_task['private']['evaluation']['parameters']['revision'],
                'scene_model': geometry_audit(new_task)['scene_model'],
                'task_sha256': hashlib.sha256(rendered[Path(files['task_file'])]).hexdigest(),
                'scene_sha256': hashlib.sha256(rendered[Path(files['scene_file'])]).hexdigest(),
                'source_identity': task['source']['json'], **files, **provenance,
                'geometry': geometry_audit(new_task, enforce=True), **FLAGS})
    for row in trays:
        task, scene = read_pair(row)
        for key in ('task', 'scene'):
            snapshots[row[key + '_file']] = digest(row[key + '_file'])
        tray_rows.append(tray_audit(row, task))
    audit = {'schema': 'wia_control_family_audit_v1', 'dimensional_pairs': audit_rows,
        'tray_families': tray_rows, 'input_sha256': snapshots,
        'risks': ['Geometry variants are alternatives; g1 and g2 must not be combined.',
            'D2 width is reported from oriented footprints; only its initialization policy is revised.',
            'T38/T39 use fixed destinations and do not invoke the collection tray initializer.',
            'No native geometry, initialization, public motion or physical success is certified.'], **FLAGS}
    manifest = {'schema': 'wia_variant_manifest_v1', 'artifact_path_base': 'absolute',
        'status': 'pending_native_qualification', 'entries': entries, 'full_reference_execution': 'pending', **FLAGS}
    design = {'schema': 'wia_control_family_geometry_design_v1', 'scope': sorted(DIMENSIONAL_ALIASES),
        'tray_audit_scope': sorted(TRAY_ALIASES), 'geometry_variants': {
            INITIALIZATION_VARIANT: 'Original geometry with initialization-only native empty-finger open policy.',
            'g1_closer': 'Shift both movable candidate centers -0.10 m along approach X only.',
            'g2_front_crop': 'Preserve far X face; crop front 0.20 m, center +0.10 m, front legs +0.20 m.',
            'g3_narrow': 'Set lateral width to 1.20 m; preserve center/depth and original leg edge insets.'},
        'minimum_support_margin_m': MARGIN,
        'initialization_policy_scope': ('Only new dimensional variants initialize both empty hands fully open. '
            'The unchanged aggregate dimensional criterion requires the working gripper fully open and both hands empty; '
            'it does not require the inactive finger aperture fully open. Omitted policy preserves legacy behavior.'),
        'tray_probe_design': {'first_candidate': 'tray_candidates/p1.json',
            'conditional_candidates_only_if_needed': ['tray_candidates/p2.json', 'tray_candidates/p3.json'],
            'commands_file': 'tray_candidates/public_commands.json',
            'scope': 'Short unloaded tray initialization and retention probes; doorway and loaded carrying remain unqualified.',
            'family_pose_activated': False},
        'input_sha256': snapshots, **FLAGS}
    for filename, value in (('audit.json', audit), ('manifest.json', manifest), ('design.json', design)):
        rendered[output_dir / filename] = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    for name, position in (('p1', [.45, -.25, .85]), ('p2', [.50, -.25, .85]), ('p3', [.45, -.30, .90])):
        candidate = {'case_id': name, 'robot_yaw_deg': 0., 'right_eef_position_base_m': position,
            'rim_axis': 'y', 'rim_penetration_m': .025, 'maximum_plate_tilt_deg': 10.}
        rendered[output_dir / 'tray_candidates' / (name + '.json')] = (json.dumps(candidate, indent=2) + '\n').encode()
    commands = [{'action': token, 'distance_m': distance} for token, distance in (
        ('TRUNK_DOWN', .08), ('TRUNK_DOWN', .08), ('TRUNK_UP', .08), ('TRUNK_UP', .08),
        ('BASE_BACK', .05), ('BASE_FWD', .05))]
    rendered[output_dir / 'tray_candidates' / 'public_commands.json'] = (json.dumps(commands, indent=2) + '\n').encode()
    return manifest, audit, rendered


def validate_outputs(manifest):
    for row in manifest['entries']:
        task = load_task(row['task_file'], variant='adapted', run_mode='diagnostic')
        scene = load_scene(row['scene_file'], variant='adapted', run_mode='diagnostic')
        if task.definition.scene != scene.definition.scene or task.seed != scene.seed:
            raise ValueError('Diagnostic loaders disagree about generated pair.')
        task.verify_unchanged()
        scene.verify_unchanged()


def _replace_tray_pose(value, position):
    """Replace only explicitly named tray EEF pose copies; return their count."""
    count = 0
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'right_eef_position_base_m':
                value[key] = deepcopy(position)
                count += 1
            else:
                count += _replace_tray_pose(item, position)
    elif isinstance(value, list):
        count += sum(_replace_tray_pose(item, position) for item in value)
    return count


def build_tray_p1_package(output_dir=DEFAULT_OUTPUT, probe_file=TRAY_P1_PROBE):
    """Build the five reviewed held-tray parents with only P1 pose replacement."""
    output_dir, probe_file = Path(output_dir).resolve(), Path(probe_file).resolve()
    probe = json.loads(probe_file.read_text())
    expected_commands = [{'action': token, 'distance_m': distance} for token, distance in (
        ('TRUNK_DOWN', .08), ('TRUNK_DOWN', .08), ('TRUNK_UP', .08), ('TRUNK_UP', .08),
        ('BASE_BACK', .05), ('BASE_FWD', .05))]
    if (probe.get('accepted') is not True or probe.get('initialization_accepted') is not True
            or probe.get('public_motion_probes_passed') is not True
            or probe['design']['right_eef_position_base_m'] != TRAY_P1_POSE
            or [row['command'] for row in probe['public_commands']] != expected_commands
            or any(row['result']['success'] is not True for row in probe['public_commands'])):
        raise ValueError('P1 variants require the exact retained unloaded P1 probe evidence.')
    citation = {'file': str(probe_file), 'sha256': digest(probe_file),
        'scope': 'P1 initialization and six short unloaded public motions only; full scene/task, doorway and loaded carrying remain unqualified.'}
    rows = manifest_rows(TRAY_MANIFESTS, TRAY_P1_ALIASES)
    rows = [r for r in rows if r['document_alias'] != 'T07' or r['manifest_file'] == str(TRAY_MANIFESTS[-1])]
    if {r['document_alias'] for r in rows} != TRAY_P1_ALIASES or len(rows) != 5:
        raise ValueError('P1 requires exactly the five reviewed held-tray parents.')
    rendered, entries = {}, []
    for row in rows:
        parent_task, parent_scene = read_pair(row)
        tray_audit(row, parent_task)
        task, scene = deepcopy(parent_task), deepcopy(parent_scene)
        # T07 also retains its historical unqualified preview variant, whose
        # four pose copies must stay synchronized with the adapted variant.
        count = _replace_tray_pose(task['scene'], TRAY_P1_POSE)
        count += _replace_tray_pose(task['private']['evaluation'], TRAY_P1_POSE)
        if count != (10 if row['document_alias'] == 'T07' else 6):
            raise ValueError('Unexpected held-tray pose-copy count: ' + row['document_alias'])
        variant_id = row['document_alias'].lower() + '_tray_p1_v1'
        provenance = {'parent_task_file': row['task_file'], 'parent_scene_file': row['scene_file'],
            'parent_task_sha256': digest(row['task_file']), 'parent_scene_sha256': digest(row['scene_file']),
            'parent_manifest_file': row['manifest_file'], 'parent_manifest_sha256': row['manifest_sha256']}
        task.update(id=parent_task['id'] + '__' + variant_id, version=parent_task['version'] + 1, status='pending')
        task['repair'].update(deepcopy(provenance), document_alias=row['document_alias'], variant_id=variant_id,
            tray_pose_candidate='P1', unloaded_tray_probe=deepcopy(citation),
            full_reference_execution='pending', full_task_qualified=False, **FLAGS)
        for key, value in task['repair'].items():
            if key.startswith('native_') and isinstance(value, bool):
                task['repair'][key] = False
        scene.update(id=parent_scene['id'] + '__' + variant_id, version=parent_scene['version'] + 1,
            status='pending', scene=deepcopy(task['scene']))
        CategoryTaskDefinition.model_validate(task)
        SceneDefinition.model_validate(scene)
        tray_audit(row, task)
        files = {}
        for kind, definition in (('task', task), ('scene', scene)):
            path = output_dir / ('tray_p1_' + kind + 's') / (variant_id + '.yaml')
            rendered[path] = yaml.safe_dump(definition, sort_keys=False, allow_unicode=False, width=88).encode()
            files[kind + '_file'] = str(path)
        entries.append({'document_alias': row['document_alias'], 'variant_id': variant_id,
            'variant': 'adapted', 'seed': task['defaults']['seed'], 'task_id': task['id'],
            'task_version': task['version'], 'scene_id': scene['id'],
            'source_identity': task['source']['json'], 'right_eef_position_base_m': TRAY_P1_POSE,
            'unloaded_tray_probe': deepcopy(citation), 'full_task_qualified': False,
            **files, **provenance, **FLAGS})
    manifest = {'schema': 'wia_variant_manifest_v1', 'artifact_path_base': 'absolute',
        'status': 'pending_full_scene_native_qualification', 'entries': entries,
        'unloaded_tray_probe': citation, 'full_reference_execution': 'pending', **FLAGS}
    rendered[output_dir / 'tray_p1_manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode()
    return manifest, rendered


def _drop_unselected_tolerance(value):
    """Remove the retired D1-only constraint from every duplicated parameter map."""
    count = 0
    if isinstance(value, dict):
        if 'unselected_position_tolerance_m' in value:
            del value['unselected_position_tolerance_m']
            count += 1
        for item in value.values():
            count += _drop_unselected_tolerance(item)
    elif isinstance(value, list):
        count += sum(_drop_unselected_tolerance(item) for item in value)
    return count


def _public_scalar_block(text, key):
    """Extract a complete public YAML scalar, including its original line wraps."""
    lines = text.splitlines(keepends=True)
    public = lines.index('public:\n')
    start = next(i for i in range(public + 1, len(lines)) if lines[i].startswith('  ' + key + ':'))
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or lines[end].startswith('    ')):
        end += 1
    return ''.join(lines[start:end])


def build_package_v2(output_dir=DEFAULT_V2_OUTPUT, parent_manifest=DEFAULT_OUTPUT / 'manifest.json'):
    """Version all existing D1 geometries while leaving D2/v1 artifacts intact."""
    output_dir, parent_manifest = Path(output_dir).resolve(), Path(parent_manifest).resolve()
    parent = json.loads(parent_manifest.read_text())
    rendered, entries, audit_rows = {}, [], []
    inputs = {str(parent_manifest): digest(parent_manifest)}
    for row in parent['entries']:
        if row['document_alias'].startswith('D2-'):
            if row['geometry_variant'] != INITIALIZATION_VARIANT:
                raise ValueError('Only the existing unchanged D2 i0_open references are reviewed.')
            entries.append(deepcopy(row))
            for kind in ('task', 'scene'):
                inputs[row[kind + '_file']] = digest(row[kind + '_file'])
            continue
        if row['document_alias'] not in {'D1-1', 'D1-2', 'D1-3'}:
            raise ValueError('The dimensional v2 manifest must contain only reviewed D1/D2 aliases.')
        parent_task, parent_scene = read_pair(row)
        if parent_task['private']['evaluation']['operation'] != 'select_pass_aperture':
            raise ValueError('Only D1 physical aperture tasks receive the free-attempt policy.')
        dimensional_layout(parent_task)
        task, scene = deepcopy(parent_task), deepcopy(parent_scene)
        constraints = task['public']['constraints']
        if constraints.count(D1_COMMITMENT_CONSTRAINT) != 1:
            raise ValueError('The exact reviewed public first-grasp/unselected constraint is required.')
        task['public']['constraints'] = [c for c in constraints if c != D1_COMMITMENT_CONSTRAINT]
        task['public']['constraints'].extend(D1_FREE_ATTEMPT_CONSTRAINTS)
        removed_task = _drop_unselected_tolerance(task)
        removed_scene = _drop_unselected_tolerance(scene)
        if removed_task < 1:
            raise ValueError('The D1 parent must contain its historical unselected position constraint.')
        provenance = {'parent_task_file': row['task_file'], 'parent_scene_file': row['scene_file'],
            'parent_task_sha256': digest(row['task_file']), 'parent_scene_sha256': digest(row['scene_file']),
            'parent_manifest_file': str(parent_manifest), 'parent_manifest_sha256': digest(parent_manifest)}
        suffix = '__free_attempts_v2'
        task.update(id=parent_task['id'] + suffix, version=parent_task['version'] + 1, status='pending')
        task['repair'].update(deepcopy(provenance), d1_evaluation_policy='done_only_free_attempts',
            scoring_version='wia_dimensional_progress_v2', completion_item_meaning=deepcopy(D1_COMPLETION_MEANING),
            completion_evaluation='Final evaluation only on explicit DONE; free attempts and candidate switching are allowed.',
            original_physical_goal_preserved=True, full_reference_execution='pending', **FLAGS)
        for key, value in task['repair'].items():
            if key.startswith('native_') and isinstance(value, bool):
                task['repair'][key] = False
        scene.update(id=parent_scene['id'] + suffix, version=parent_scene['version'] + 1,
            status='pending', scene=deepcopy(task['scene']))
        CategoryTaskDefinition.model_validate(task)
        SceneDefinition.model_validate(scene)
        if dimensional_layout(task) != dimensional_layout(parent_task):
            raise ValueError('Semantic revision changed physical initialization geometry.')
        files = {}
        for kind, definition in (('task', task), ('scene', scene)):
            original_path = Path(row[kind + '_file'])
            original_text = original_path.read_text()
            path = output_dir / kind / (original_path.stem + suffix + '.yaml')
            if path == original_path or output_dir == parent_manifest.parent:
                raise ValueError('D1 v2 output must be separate from every immutable v1 artifact.')
            text = yaml.safe_dump(definition, sort_keys=False, allow_unicode=False, width=88)
            if kind == 'task':
                for key in ('objective', 'success_criteria'):
                    text = text.replace(_public_scalar_block(text, key), _public_scalar_block(original_text, key), 1)
            rendered[path] = text.encode()
            files[kind + '_file'] = str(path)
            inputs[str(original_path)] = digest(original_path)
        new_row = {**deepcopy(row), **files, **provenance,
            'variant_id': row['variant_id'] + suffix, 'task_id': task['id'], 'task_version': task['version'],
            'scene_id': scene['id'], 'scene_version': scene['version'],
            'task_sha256': hashlib.sha256(rendered[Path(files['task_file'])]).hexdigest(),
            'scene_sha256': hashlib.sha256(rendered[Path(files['scene_file'])]).hexdigest(),
            'd1_evaluation_policy': 'done_only_free_attempts', 'scoring_version': 'wia_dimensional_progress_v2',
            'completion_item_meaning': deepcopy(D1_COMPLETION_MEANING), **FLAGS}
        entries.append(new_row)
        audit_rows.append({'document_alias': row['document_alias'], 'seed': row['seed'],
            'geometry_variant': row['geometry_variant'], **files, **provenance,
            'removed_unselected_parameter_copies': {'task': removed_task, 'scene': removed_scene},
            'objective_yaml_block_byte_preserved': True, 'success_criteria_yaml_block_byte_preserved': True,
            'physical_layout_unchanged': True, 'completion_definitions_unchanged': True})
    if len(audit_rows) != 48 or len(entries) - len(audit_rows) != 8:
        raise ValueError('The reviewed v1 package requires 48 D1 revisions and 8 unchanged D2 references.')
    tray_manifest = DEFAULT_OUTPUT / 'tray_p1_manifest.json'
    if tray_manifest.is_file():
        inputs[str(tray_manifest)] = digest(tray_manifest)
    manifest = {'schema': 'wia_variant_manifest_v1', 'artifact_path_base': 'absolute',
        'revision': 'control_family_repairs_v2', 'status': 'pending_native_qualification',
        'parent_manifest_file': str(parent_manifest), 'parent_manifest_sha256': digest(parent_manifest),
        'active_dimensional_entry': True, 'new_d1_semantic_versions': 48, 'unchanged_d2_references': 8,
        'd1_evaluation_policy': 'done_only_free_attempts', 'd1_scoring_version': 'wia_dimensional_progress_v2',
        'd1_completion_item_meaning': deepcopy(D1_COMPLETION_MEANING),
        'entries': entries, 'full_reference_execution': 'pending', **FLAGS}
    design = {'schema': 'wia_control_family_semantic_design_v2',
        'policy': 'D1 free attempts and switching; final evaluation only on DONE.',
        'preserved_goal': 'Real grasp/lift, actual framed aperture traversal, supported blue-pad placement, release and stability.',
        'scope': 'All 48 existing D1 variants only; D2 and P1 tray definitions remain unchanged.',
        'parent_manifest_file': str(parent_manifest), 'input_sha256': inputs, **FLAGS}
    audit = {'schema': 'wia_control_family_semantic_audit_v2', 'entries': audit_rows, 'input_sha256': inputs,
        'new_d1_versions': 48, 'unchanged_d2_references': 8, **FLAGS}
    for filename, value in (('manifest.json', manifest), ('design.json', design), ('audit.json', audit)):
        rendered[output_dir / filename] = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    return manifest, audit, rendered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--revision', choices=('v1', 'v2'), default='v2')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--include-tray-p1', action='store_true', help='Also build the five reviewed P1 held-tray definitions.')
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = DEFAULT_V2_OUTPUT if args.revision == 'v2' else DEFAULT_OUTPUT
    if args.revision == 'v2' and args.include_tray_p1:
        raise ValueError('P1 tray v1 artifacts remain valid; use their existing manifest unchanged.')
    manifest, audit, rendered = (build_package_v2(args.output_dir) if args.revision == 'v2'
        else build_package(args.output_dir))
    tray_manifest = None
    if args.include_tray_p1:
        tray_manifest, tray_rendered = build_tray_p1_package(args.output_dir)
        rendered.update(tray_rendered)
    for path, value in rendered.items():
        if path.exists() and path.read_bytes() != value:
            raise ValueError('Refusing to overwrite differing output: ' + str(path))
        if args.check and not path.is_file():
            raise ValueError('Missing output: ' + str(path))
        if not args.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
    validate_outputs(manifest)
    if tray_manifest:
        validate_outputs(tray_manifest)
    for path, expected in audit['input_sha256'].items():
        if digest(path) != expected:
            raise ValueError('Input changed during build: ' + path)
    if args.revision == 'v2':
        print(json.dumps({'new_d1_semantic_versions': 48, 'unchanged_d2_references': 8,
            'native_validation_run': False, 'output_dir': str(args.output_dir.resolve())}))
        return 0
    print(json.dumps({'dimensional_pairs_audited': len(audit['dimensional_pairs']),
        'tray_families_audited': len(audit['tray_families']),
        'geometry_candidates': sum(r['geometry_variant'] in VARIANTS for r in manifest['entries']),
        'initialization_only_candidates': sum(r['geometry_variant'] == INITIALIZATION_VARIANT for r in manifest['entries']),
        'held_tray_p1_candidates': len(tray_manifest['entries']) if tray_manifest else 0,
        'native_validation_run': False, 'output_dir': str(args.output_dir.resolve())}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
