"""Build and calibrate source-sized joint capacity inventories.

Run CPU checks: python -B -m pytest -q tests/test_task_edit2_capacity.py
Native calibration is diagnostic-only: supply an isolated trial context manager
and an explicit full-inventory reset callback before starting any Agent episode.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from wia.authoring.converters.composition_manipulation import capacity_order_graph

REVISION = 'task_edit2_capacity_inventory_20261018_v1'
OPERATION = 'report_joint_capacity_ranking'
INITIALIZATION = 'joint_source_capacity_inventory_v1'
from wia.resources import ROOT, ESI_ROOT
SOURCE_PROTOCOL = ESI_ROOT / 'src/dataset_generation/task_capacity/batch_pour.py'
TIE_RULE = 'Equal retained particle counts within the same fill_both trial; no cross-trial count comparison.'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def asset_fingerprint(item):
    return hashlib.sha256(_json({key: item[key] for key in ('category', 'model', 'scale')}).encode()).hexdigest()


def build_inventory(source_evidence):
    """Union asset identities/scales, retaining every source-side remapping.

    Evidence consists of (SourceInstance, independently checked capacity_evidence)
    pairs. Only same-trial relations transfer; particle counts from distinct
    records are never subtracted or sorted against one another.
    """
    from wia.authoring.converters.physical_structure_liquid_volume import _source_data
    inventory, by_asset, mappings, comparisons = [], {}, [], []
    scene_room = None
    for source, evidence in sorted(source_evidence, key=lambda row: row[0].relative_path):
        raw, payload, entries, _ = _source_data(source)
        current = (raw['scene'], raw['room'])
        if scene_room is None:
            scene_room = current
        if current != scene_room or current != ('Rs_int', 'living_room_0'):
            raise ValueError('Joint capacity inventory requires compatible Rs_int living-room sources.')
        if not evidence.get('valid') or evidence.get('source_identity') != source.relative_path:
            raise ValueError('Every source pair requires valid same-trial identity and scale evidence.')
        remap = {}
        for side in ('left', 'right'):
            scale = evidence['expected_native_scales'][side]
            scale = [float(scale)] * 3 if isinstance(scale, (float, int)) else list(scale)
            item = {'category': entries[side]['category'], 'model': entries[side]['model'], 'scale': scale}
            fingerprint = asset_fingerprint(item)
            if fingerprint not in by_asset:
                index = len(inventory)
                item.update(id=chr(ord('A') + index), binding=f'container_{index:02d}',
                    name=f'capacity_container_{index:02d}', source_occurrences=[])
                by_asset[fingerprint] = item
                inventory.append(item)
            item = by_asset[fingerprint]
            occurrence = {'source_identity': source.relative_path, 'side': side,
                'source_name': 'obj1' if entries[side].get('side') == 'left' else 'obj2'}
            item['source_occurrences'].append(occurrence)
            remap[side] = item['id']
        if remap['left'] == remap['right']:
            raise ValueError('A source pair cannot collapse to one physical container.')
        order = evidence.get('expected_order', '').split()
        if len(order) != 3 or order[0] not in remap or order[2] not in remap:
            raise ValueError('Source capacity relation cannot be remapped by identity.')
        comparisons.append({'first': remap[order[0]], 'relation': order[1], 'second': remap[order[2]],
            'origin': 'source_same_trial', 'source_identity': source.relative_path,
            'source_sha256': hashlib.sha256(source.path.read_bytes()).hexdigest(),
            'counts': {remap[side]: evidence['counts'][side] for side in ('left', 'right')},
            'asset_fingerprints': {item_id: asset_fingerprint(next(item for item in inventory if item['id'] == item_id))
                for item_id in remap.values()}, 'source_evidence': copy.deepcopy(evidence)})
        mappings.append({'source_identity': source.relative_path,
            'source_sha256': hashlib.sha256(source.path.read_bytes()).hexdigest(), 'sides': remap})
    if not inventory:
        raise ValueError('A joint capacity inventory cannot be empty.')
    # The immutable source constructor/postprocessor resolves these two names.
    # All public/evaluator bindings are still namespaced and remapped explicitly.
    anchor = mappings[0]['sides']
    for side, name in (('left', 'obj1'), ('right', 'obj2')):
        next(item for item in inventory if item['id'] == anchor[side])['name'] = name
    receipt = {'schema': REVISION, 'scene_model': scene_room[0], 'room': scene_room[1],
        'inventory': inventory, 'source_mappings': mappings, 'comparisons': comparisons,
        'tie_rule': TIE_RULE, 'cross_trial_particle_counts_comparable': False,
        'source_protocol_sha256': hashlib.sha256(SOURCE_PROTOCOL.read_bytes()).hexdigest()}
    receipt['inventory_id'] = hashlib.sha256(_json(inventory).encode()).hexdigest()[:24]
    receipt['truth'] = validate_truth(receipt)
    receipt['calibration_recipe'] = {'status': 'pending', 'diagnostic_only': True,
        'protocol': str(SOURCE_PROTOCOL), 'functions': ['fill_both', 'get_particles_in_obj'],
        'required_pairs': receipt['truth'].get('missing_pairs', []),
        'isolated_reset_per_pair': True, 'identical_water_settings_and_particle_scales': True,
        'preserve_all_asset_scales': True, 'reset_entire_inventory_before_agent_episode': True,
        'agent_operation_count': 0, 'tie_rule': TIE_RULE}
    return receipt


def validate_truth(receipt):
    """Accept only identity-matched same-trial source/native measurement edges."""
    items = {item['id']: item for item in receipt['inventory']}
    accepted, invalid = [], []
    native_settings = None
    protocol_hash = receipt['source_protocol_sha256']
    for row in receipt.get('comparisons', []):
        first, second = row.get('first'), row.get('second')
        ids = (first, second)
        reason = None
        if first not in items or second not in items or first == second or row.get('relation') not in ('<', '=', '>'):
            reason = 'invalid_capacity_edge'
        elif any(row.get('asset_fingerprints', {}).get(key) != asset_fingerprint(items[key]) for key in ids):
            reason = 'capacity_asset_identity_or_scale_mismatch'
        elif any(type(row.get('counts', {}).get(key)) is not int or row['counts'][key] <= 0 for key in ids):
            reason = 'capacity_same_trial_counts_missing'
        elif row.get('relation') != ('=' if row['counts'][first] == row['counts'][second] else
                '<' if row['counts'][first] < row['counts'][second] else '>'):
            reason = 'capacity_relation_count_conflict'
        elif row.get('origin') == 'source_same_trial':
            evidence = row.get('source_evidence', {})
            mapping = next((entry for entry in receipt.get('source_mappings', [])
                if entry.get('source_identity') == row.get('source_identity')), {})
            sides = mapping.get('sides', {})
            order = evidence.get('expected_order', '').split()
            mapped_order = [sides.get(order[0]), order[1], sides.get(order[2])] if len(order) == 3 else None
            mapped_counts = {key: evidence.get('counts', {}).get(side) for side, key in sides.items()}
            scales_match = all(np.allclose(items[key]['scale'], evidence.get('expected_native_scales', {}).get(side, np.nan),
                atol=1e-6, rtol=1e-5) for side, key in sides.items() if key in items)
            if (not evidence.get('valid') or evidence.get('source_identity') != row.get('source_identity') or
                    evidence.get('source_protocol', {}).get('sha256') != protocol_hash or
                    evidence.get('calibration', {}).get('same_water_system_within_trial') is not True or
                    mapped_order != [first, row['relation'], second] or mapped_counts != row['counts'] or
                    mapping.get('source_sha256') != row.get('source_sha256') or not scales_match):
                reason = 'source_same_trial_protocol_unverified'
        elif row.get('origin') == 'native_common_protocol':
            settings = row.get('water_settings')
            reset = receipt.get('post_calibration_inventory_reset', {})
            if (row.get('source_protocol_sha256') != protocol_hash or not settings or
                    row.get('diagnostic_only') is not True or row.get('agent_operation_count') != 0 or
                    row.get('isolated_reset_verified') is not True or not all(reset.get(key) is True
                        for key in ('accepted', 'all_inventory_restored', 'agent_history_empty'))):
                reason = 'native_common_protocol_unverified'
            elif native_settings is not None and settings != native_settings:
                reason = 'native_water_system_settings_mismatch'
            else:
                native_settings = settings
        else:
            reason = 'unsupported_capacity_evidence_origin'
        if reason:
            invalid.append({'first': first, 'second': second, 'reason': reason})
        else:
            accepted.append([first, row['relation'], second])
    graph = capacity_order_graph(list(items), accepted)
    return {**graph, 'accepted': bool(graph['accepted'] and not invalid),
        'valid_comparisons': accepted, 'invalid_evidence': invalid,
        'pair_denominator': len(items) * (len(items) - 1) // 2,
        'reason': 'invalid_capacity_evidence' if invalid else graph.get('reason')}


def load_source_fill_protocol():
    """Load original fill helpers without executing source global macro changes."""
    import torch
    import omnigibson as og
    import omnigibson.lazy as lazy
    import omnigibson.utils.transform_utils as transforms
    from omnigibson.prims.xform_prim import XFormPrim
    names = {'generate_box', 'generate_particles_in_box', 'check_in_contact', 'get_particles_in_obj', 'fill_both'}
    source = ast.parse(SOURCE_PROTOCOL.read_text())
    selected = ast.Module(body=[node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    namespace = {'th': torch, 'og': og, 'lazy': lazy, 'T': transforms, 'XFormPrim': XFormPrim}
    exec(compile(selected, str(SOURCE_PROTOCOL), 'exec'), namespace)
    return namespace


def native_water_settings(water):
    """Record actual particle geometry, scale range and density, not count totals."""
    result = {'system_name': water.name, 'system_class': type(water).__name__}
    for field in ('particle_radius', 'particle_contact_radius', 'min_scale', 'max_scale', 'particle_density'):
        value = getattr(water, field)
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=float)
        if not np.isfinite(array).all() or np.any(array <= 0):
            raise ValueError('Native water settings are unavailable or nonpositive.')
        result[field] = array.tolist()
    if result['min_scale'] != result['max_scale']:
        raise ValueError('Common capacity calibration requires fixed identical particle scales.')
    return result


def calibrate_native_pairs(receipt, diagnostic_trial, reset_actual_inventory, *, pairs=None, fill_protocol=None):
    """Run original fill_both in isolated native diagnostic contexts, then reset.

    ``diagnostic_trial(first_item, second_item)`` is a context manager returning
    ``objects`` (two native objects), ``water``, ``floor_z`` (two source bottoms
    + 0.05 m) and ``isolated_reset_verified=True``. It owns trial environment
    creation and destruction, original source pair placement (first at left),
    source joint initialization and pool cleanup. It must never yield the live
    Agent episode. ``reset_actual_inventory`` returns an accepted full reset
    receipt with ``all_inventory_restored`` and ``agent_history_empty`` true.
    The helper itself performs diagnostic simulator setters only via the
    unchanged source fill helpers. Agent controls are never invoked.
    """
    import torch
    calibrated = copy.deepcopy(receipt)
    items = {item['id']: item for item in receipt['inventory']}
    pairs = pairs if pairs is not None else validate_truth(receipt).get('missing_pairs', [])
    settings = None
    failure = None
    try:
        protocol = fill_protocol or load_source_fill_protocol()
        if hashlib.sha256(SOURCE_PROTOCOL.read_bytes()).hexdigest() != receipt['source_protocol_sha256']:
            raise ValueError('The native source fill protocol differs from the versioned receipt.')
        for first, second in pairs:
            if first not in items or second not in items or first == second:
                raise ValueError('Calibration pair is not in the explicit inventory.')
            with diagnostic_trial(items[first], items[second]) as trial:
                if trial.get('isolated_reset_verified') is not True or trial.get('agent_episode_active', True):
                    raise ValueError('Capacity calibration must use a separate diagnostic reset.')
                objects, water = trial['objects'], trial['water']
                for item, obj in zip((items[first], items[second]), objects):
                    scale = obj.scale.detach().cpu().numpy() if hasattr(obj.scale, 'detach') else obj.scale
                    if (obj.category != item['category'] or obj.model != item['model'] or
                            not np.allclose(scale, item['scale'], atol=1e-6, rtol=1e-5)):
                        raise ValueError('Diagnostic container identity or native scale differs from source.')
                current_settings = native_water_settings(water)
                if settings is not None and current_settings != settings:
                    raise ValueError('Water parameters or particle scale changed across diagnostic trials.')
                settings = current_settings
                offsets = torch.stack([axis * sign * water.particle_radius for axis in torch.eye(3)
                    for sign in (-1, 1)] + [torch.zeros(3)])
                counts = protocol['fill_both'](*objects, water, *trial['floor_z'], offsets)
                measured = [protocol['get_particles_in_obj'](obj, water, offsets)[2] for obj in objects]
                if tuple(int(mask.sum().item()) for mask in measured) != tuple(counts):
                    raise ValueError('Source fill and retained-particle measurement disagree.')
                a, b = map(int, counts)
                calibrated['comparisons'].append({'first': first, 'relation': '=' if a == b else '<' if a < b else '>',
                    'second': second, 'origin': 'native_common_protocol', 'counts': {first: a, second: b},
                    'asset_fingerprints': {key: asset_fingerprint(items[key]) for key in (first, second)},
                    'water_settings': current_settings, 'source_protocol_sha256': receipt['source_protocol_sha256'],
                    'diagnostic_only': True, 'agent_operation_count': 0, 'isolated_reset_verified': True})
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
    finally:
        try:
            reset = reset_actual_inventory()
        except Exception as error:
            reset = {'accepted': False, 'all_inventory_restored': False, 'agent_history_empty': False,
                'error': f'{type(error).__name__}: {error}'}
        calibrated['post_calibration_inventory_reset'] = reset
    calibrated['truth'] = validate_truth(calibrated)
    reset_ok = all(reset.get(key) is True for key in ('accepted', 'all_inventory_restored', 'agent_history_empty'))
    if failure or not reset_ok:
        calibrated['truth'].update(accepted=False, reason='native_calibration_or_actual_inventory_reset_failed')
    calibrated['calibration_recipe'].update(status='complete' if calibrated['truth']['accepted'] else 'pending',
        native_error=failure, post_calibration_reset_verified=reset_ok, water_settings=settings)
    return calibrated


def build_joint_candidate(source_evidence, context, *, calibration_receipt=None):
    """Create one provisional physical task and scene, never a row-only merge."""
    from wia.authoring.conversion import original_scene, physical_task
    from wia.authoring.converters.physical_structure_liquid_volume import MODULE, FLOOR_NAME
    receipt = build_inventory(source_evidence)
    if calibration_receipt is not None:
        if any(calibration_receipt.get(key) != receipt[key] for key in (
                'inventory_id', 'inventory', 'source_mappings', 'source_protocol_sha256')) or not all(
                    row in calibration_receipt.get('comparisons', []) for row in receipt['comparisons']):
            raise ValueError('Native calibration receipt differs from the exact source inventory/protocol.')
        receipt = copy.deepcopy(calibration_receipt)
        receipt['truth'] = validate_truth(receipt)
    source_by_identity = {source.relative_path: source for source, _ in source_evidence}
    anchor = source_by_identity[receipt['source_mappings'][0]['source_identity']]
    inventory = receipt['inventory']
    keys = [item['binding'] for item in inventory]
    bindings = {item['binding']: {key: item[key] for key in ('name', 'category', 'model')} for item in inventory}
    bindings['floor'] = {'name': FLOOR_NAME}
    initialization = {'operation': INITIALIZATION, 'inventory_receipt': receipt}
    policy = {'mode': 'inspection', 'group_bindings': keys, 'manipulation_bindings': [],
        'relocation_policy': 'shared_tabletop', 'placement_groups': [[key] for key in keys],
        'support_bindings': keys, 'source_camera_policy': 'preserve',
        'source_region': receipt['room'], 'support_search': 'measured_surfaces',
        'allow_fallback_table': True, 'minimum_grasp_height_m': .35,
        'maximum_grasp_height_m': 1.10, 'maximum_stance_distance_m': .85}
    patches = [{'operation': 'category_initialization', 'phase': 'before_construction',
        'module': MODULE, 'parameters': initialization}, {'operation': 'add_existing_r1'},
        {'operation': 'category_initialization', 'phase': 'after_postprocessing',
            'module': MODULE, 'parameters': initialization},
        {'operation': 'prepare_category_interaction', **policy}]
    public = {'objective': 'Compare all labeled containers on the shared table and report increasing '
        'capacity order, including ties. Use the visible letter labels; for example, A < B = C. '
        'Capacity means the amount of water a source-sized container can hold.',
        'success_criteria': 'Report every pair correctly for an exact ranking. Comparison accuracy is '
            '100 times correct pair relations divided by n(n-1)/2. Omitted or malformed pairs receive '
            'no credit. Equal capacity uses equal retained particle counts under the same fill protocol.',
        'environment': 'All containers are present together on one table. ' + '; '.join(
            f"{item['id']}: {item['category'].replace('_', ' ')}" for item in inventory) +
            '. Letter plates identify each physical container as you change viewpoint.',
        'constraints': ['Use the provided observations and guarded robot controls to inspect the containers.',
            'Preserve the containers, their source scales and labels. No pouring or physical rearrangement is required.',
            'Do not substitute current liquid amount, fill height, fill fraction or exterior bounding-box volume for capacity.']}
    parameters = {'task_revision': REVISION, 'inventory_receipt': receipt,
        'object_keys': keys, 'pair_denominator': len(inventory) * (len(inventory) - 1) // 2,
        'completion_items': [{'id': f"capacity:{first['id']}:{second['id']}", 'kind': 'semantic_answer'}
            for index, first in enumerate(inventory) for second in inventory[index + 1:]],
        'native_label_visibility_verification_required': True}
    evaluation = {'type': 'category_operation', 'module': MODULE, 'operation': OPERATION,
        'binding_keys': [*keys, 'floor'], 'parameters': parameters}
    task = physical_task(anchor, context, public=public, evaluation=evaluation,
        bindings=bindings, patches=patches, capabilities=['base_translation', 'base_rotation'],
        qualification={'on_failure': 'skip', 'initial_visibility': 'none', 'target_movable_collidable': False})
    task['id'] = 'task_capacity_union_' + receipt['inventory_id']
    task['defaults']['max_steps'] = 160
    task['repair'] = {'task_revision': REVISION, 'mechanism_contract': {
        'joint_shared_reset_required': True, 'source_scales_preserved': True,
        'goal': 'capacity_comparison_only', 'cross_trial_count_inference_forbidden': True}}
    scene = original_scene(anchor, context, bindings=bindings, patches=patches)
    scene['id'] = 'scene_capacity_union_' + receipt['inventory_id']
    pending = ['single_table_native_fit', 'robot_upright_floor_support_and_access', 'observable_label_visibility']
    if not receipt['truth']['accepted']:
        pending.append('complete_common_protocol_capacity_truth')
    return {'candidate_id': 'capacity_union_' + receipt['inventory_id'], 'status': 'pending',
        'task_revision': REVISION, 'source_identity': anchor.relative_path,
        'source_mappings': receipt['source_mappings'], 'inventory_receipt': receipt,
        'task_definition': task, 'scene_definition': scene, 'interaction_policy': policy,
        'native_execution_verified': False, 'pending_gates': pending}


def attach_native_labels(session, receipt):
    """Place rendered letter plates next to bound fixed containers, without collision.

    Association is known from construction; mounted-sensor readability still
    requires native review and must not be inferred from the existence of a mesh.
    Textures are reproducible temporary runtime assets, not source modifications.
    """
    from PIL import Image, ImageDraw, ImageFont
    from pxr import Gf, Sdf, UsdGeom, UsdShade
    stage = session.pipeline.og.sim.stage
    cache = Path('/tmp') / ('capacity_labels_' + receipt['inventory_id'])
    cache.mkdir(parents=True, exist_ok=True)
    robot_position = np.asarray(session.robot.get_position_orientation()[0].cpu(), dtype=float)
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
    font = ImageFont.truetype(str(font_path), 160) if font_path.is_file() else ImageFont.load_default()
    rows = {}
    for item in receipt['inventory']:
        obj = session.env.scene.object_registry('name', item['name'])
        low, high = [np.asarray(value.cpu(), dtype=float) for value in obj.aabb]
        center = (low + high) / 2
        horizontal = robot_position[:2] - center[:2]
        horizontal /= np.linalg.norm(horizontal)
        tangent = np.array([-horizontal[1], horizontal[0], 0.])
        bottom = np.r_[center[:2] + horizontal * .5 * np.linalg.norm(high[:2] - low[:2]), low[2] + .003]
        points = [bottom - .04 * tangent, bottom + .04 * tangent,
            bottom + .04 * tangent + [0., 0., .08], bottom - .04 * tangent + [0., 0., .08]]
        texture = cache / (item['id'] + '.png')
        bitmap = Image.new('RGB', (256, 256), 'white')
        draw = ImageDraw.Draw(bitmap)
        draw.text((128, 128), item['id'], font=font, fill='black', anchor='mm')
        bitmap.save(texture)
        path = f"/World/CapacityLabels/{item['binding']}"
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr([Gf.Vec3f(*map(float, point)) for point in points])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateSubdivisionSchemeAttr('none')
        uv = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        uv.Set([(0., 0.), (1., 0.), (1., 1.), (0., 1.)])
        material = UsdShade.Material.Define(stage, path + '/Material')
        shader = UsdShade.Shader.Define(stage, path + '/Material/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        sampler = UsdShade.Shader.Define(stage, path + '/Material/Texture')
        sampler.CreateIdAttr('UsdUVTexture')
        sampler.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(str(texture))
        reader = UsdShade.Shader.Define(stage, path + '/Material/UV')
        reader.CreateIdAttr('UsdPrimvarReader_float2')
        reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
        sampler.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(sampler.ConnectableAPI(), 'rgb')
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(1.)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)
        rows[item['id']] = {'object_name': item['name'], 'binding': item['binding'],
            'label_prim_path': path, 'texture_path': str(texture), 'visible_mesh_created': True,
            'mounted_sensor_readability_verified': False}
    return rows
