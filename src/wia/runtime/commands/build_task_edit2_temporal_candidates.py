"""Generate source-supported TaskEdit2 temporal definitions without native execution.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B \
  src/wia/runtime/commands/build_task_edit2_temporal_candidates.py --output outputs/task_edit2_20261018/temporal
All supports, routes, dynamic preloading and action feasibility remain native pending.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

from wia.resources import ESI_ROOT as WIA_ESI_ROOT

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from wia.authoring.conversion import ConversionContext, SourceInstance
from wia.authoring.converters import temporal_understanding_agent_observation as history
from wia.authoring.converters import temporal_understanding_two_phase as change
from wia.authoring.converters import temporal_understanding_unobserved_change as source_change
from wia.runtime.scene_capture_queue import digest, sha256, write_json
from wia.tasks.loader import task_definition
from wia.tasks.schema import SceneDefinition
from wia.tasks.interaction_setup import prepare_definitions

HISTORY_SOURCE = 'Temporal Understanding/Agent Observation/Ihlen_0_int/storage_room_0/Ihlen_0_int__storage_room_0__run_000.json'
CHANGE_SOURCES = {
    'added': 'Temporal Understanding/Unobserved Change/Beechwood_0_garden/living_room_1/change_detection__q_000.json',
    'removed': 'Temporal Understanding/Unobserved Change/Beechwood_0_garden/living_room_0/current_state_reasoning__q_001.json',
    'replaced': 'Temporal Understanding/Unobserved Change/house_single_floor/entryway_0/change_identification__q_000.json',
    'no_change': 'Temporal Understanding/Unobserved Change/Beechwood_0_garden/corridor_0/change_detection__q_000.json',
}
CORE_MANIFEST = Path('/home/zhouxinyang/esi_category_conversion/core_revision_20261018/core_candidates_for_review.json')
CORE_OUTPUT = ROOT / 'outputs/task_edit2_20261018/temporal_core_v3'
CORE_HISTORY_REVISION = 'task_edit2_core_history_three_targets_v1'
CORE_HISTORY_ASSETS = (
    ('history_added_jam', 'jar_of_jam', 'vxjbjf', 'native_geometry_core_v3', 'obj3'),
    ('history_added_box', 'box_of_sanitary_napkins', 'ykpeja', 'native_deformable_v4',
        'cover_small_item_render_item_0788552322'),
)
CORE_CHANGE_BRANCHES = {
    'Temporal Understanding/Unobserved Change/house_single_floor/dining_room_0/change_detection__q_000.json': 'removed',
    'Temporal Understanding/Unobserved Change/house_single_floor/living_room_1/change_identification__q_000.json': 'added',
    'Temporal Understanding/Unobserved Change/hotel_gym_spa/gym_0/change_detection__q_001.json': 'added',
    'Temporal Understanding/Unobserved Change/office_large/bathroom_0/change_detection__q_001.json': 'no_change',
}


def source(identity, category):
    return SourceInstance(WIA_ESI_ROOT / 'dataset/json_clean' / identity, identity, 'Temporal Understanding', category)


def tray_plan(prefix, xy, floor_z):
    surface = floor_z + .72
    rows = history._table_primitives(prefix + '_table', xy, floor_z, surface, [.35, .28, .18, 1.])
    rows.append({'name': prefix + '_tray', 'category': 'tray', 'position': [*xy, surface + .01],
        'dimensions_m': [.5, .4, .02], 'rgba': [.15, .3, .9, 1.]})
    for i, (dx, dy, sx, sy) in enumerate(((-.245, 0., .01, .4), (.245, 0., .01, .4),
                                        (0., -.195, .5, .01), (0., .195, .5, .01))):
        rows.append({'name': f'{prefix}_rim_{i}', 'category': 'tray',
            'position': [xy[0] + dx, xy[1] + dy, surface + .04],
            'dimensions_m': [sx, sy, .06], 'rgba': [.15, .3, .9, 1.]})
    return rows, surface


def history_config(src, inventory=None, public_room_label=None):
    _, payload, original_objects = history._source_data(src)
    objects = original_objects if inventory is None else inventory
    # Donor asset coordinates belong to other native scenes. Anchor this room's
    # station proposal to its own original source, never to donor world XY.
    floor = float(np.median([row['aabb_min'][2] for row in original_objects]))
    middle = np.mean([row['position'][:2] for row in original_objects], axis=0)
    colors = [('red', [.85, .18, .12, 1.]), ('green', [.15, .7, .25, 1.]), ('blue', [.15, .3, .85, 1.])]
    rows, positions, supports, labels = [], {}, {}, {}
    for i, obj in enumerate(objects):
        color_name, color = colors[i]
        xy = [float(middle[0] + (i - 1) * .9), float(middle[1])]
        prefix = f'history_{color_name}_station'
        rows.extend(history._table_primitives(prefix, xy, floor, floor + .72, color))
        positions[obj['_key']] = [*xy, float(floor + .72 + obj['position'][2] - obj['aabb_min'][2] + .002)]
        supports[obj['_key']] = prefix
        labels[obj['_key']] = 'the ' + obj['category'].replace('_', ' ') + ' at the ' + color_name + ' table'
    destination_xy = [float(middle[0]), float(middle[1] - 1.1)]
    tray, _ = tray_plan('history_delivery', destination_xy, floor)
    rows.extend(tray)
    return {'task_edit2_temporal': True, 'history_target_labels': labels,
        'history_public_route': 'Survey the red, green and blue table landmarks in ' + (public_room_label or 'the storage room') + ', changing viewpoint as needed. '
            'Remember actual first sightings; the listed landmarks do not prescribe encounter order.',
        'history_destination': {'binding_key': 'history_delivery_tray', 'public_label': 'the blue-rimmed delivery tray', 'relation': 'on_top'},
        'history_apparatus': {'primitives': rows, 'source_target_positions': positions,
            'target_support_bindings': supports,
            'proposal_status': 'native_floor_support_collision_reach_and_route_pending',
            'source_placement_changed': True, 'source_models_scales_and_inventory_preserved': True,
            'floor_contact_names': [r['name'] for r in rows if '_leg_' in r['name']]}}


def core_history_config(src):
    """Propose the user-approved two measured assets beside one original target."""
    _, _, originals = history._source_data(src)
    if len(originals) != 1:
        raise ValueError('Approved Core history redesign requires its exact single original target.')
    additions = []
    for key, category, model, directory, native_name in CORE_HISTORY_ASSETS:
        receipt_path = ROOT / 'outputs/task_edit2_20261018' / directory / 'initial_state.json'
        receipt = json.loads(receipt_path.read_text())
        geometry = receipt['interaction_setup']['original_geometry'][native_name]
        native_identity = receipt['objects'][native_name]
        if native_identity['category'] != category or native_identity['scale'] != [1., 1., 1.]:
            raise ValueError('Approved added asset differs from its actual native identity/scale receipt.')
        position, low, high, quat = (np.asarray(geometry[field], dtype=float) for field in
            ('position_m', 'aabb_min_m', 'aabb_max_m', 'quaternion_xyzw'))
        if (position.shape != (3,) or low.shape != (3,) or high.shape != (3,) or quat.shape != (4,) or
                not all(np.isfinite(value).all() for value in (position, low, high, quat)) or np.any(high <= low)):
            raise ValueError('Measured donor asset pose/AABB is unavailable.')
        additions.append({'_key': key, 'category': category, 'model': model,
            'position': position.tolist(), 'quaternion_xyzw': quat.tolist(), 'scale': [1., 1., 1.],
            'aabb_min': low.tolist(), 'aabb_max': high.tolist(), 'aabb_centre': ((low + high) / 2).tolist(),
            'asset_provenance': {'source_file': str(receipt_path), 'source_sha256': sha256(receipt_path),
                'source_object_name': native_name,
                'geometry_json_path': 'interaction_setup.original_geometry.' + native_name,
                'native_identity_json_path': 'objects.' + native_name,
                'native_category': category, 'native_scale': [1., 1., 1.],
                'root_offset_from_aabb_min_m': (position - low).tolist(),
                'donor_geometry': deepcopy(geometry),
                'geometry_status': 'measured_native_donor_geometry_initialized_placement_proposal_only'}})
    config = history_config(src, [*originals, *additions], 'the ' + src.raw['room'].replace('_', ' '))
    config.update(history_inventory_revision=CORE_HISTORY_REVISION)
    dataset = []
    for row in additions:
        position = np.asarray(config['history_apparatus']['source_target_positions'][row['_key']])
        delta = position - np.asarray(row['position'])
        for field in ('aabb_min', 'aabb_max', 'aabb_centre'):
            row[field] = (np.asarray(row[field]) + delta).tolist()
        row['position'] = position.tolist()
        row['asset_provenance']['proposal_translation_m'] = delta.tolist()
        dataset.append({'type': 'DatasetObject', 'name': row['_key'], 'category': row['category'],
            'model': row['model'], 'position': row['position'], 'orientation': row['quaternion_xyzw'],
            'scale': [1., 1., 1.], 'fixed_base': False, 'visual_only': False})
    config['history_additional_targets'] = additions
    config['history_apparatus']['dataset_objects'] = dataset
    config['history_apparatus']['original_source_inventory_preserved'] = True
    config['history_apparatus']['explicit_approved_added_target_count'] = 2
    # The converter owns inventory validation; builder and converter must consume
    # precisely the same original-plus-added rows rather than infer identities.
    inventory = history.history_target_inventory(src, config)
    if len(inventory) != 3:
        raise ValueError('Approved history target inventory must contain exactly three distinct targets.')
    return config


def change_config(src):
    layout = source_change._layout(src)
    recipe = source_change.source_change_recipe(layout)
    floor = float(layout['camera']['position'][2] - 1.25)
    bounds = np.array(layout['bounds'])
    xy = [float(bounds[0] + .6), float(bounds[1] + .6)]
    rows, surface = tray_plan('change_collection', xy, floor)
    spares, dataset = [], []
    if recipe['branch'] == 'removed':
        spare_xy = [float(bounds[0] + .6), float(bounds[1] + 1.7)]
        rows.extend(history._table_primitives('change_spare_table', spare_xy, floor, floor + .72, [.8, .65, .15, 1.]))
        # Every original identity is publicly present, avoiding a missing-item hint.
        identities = []
        for key in layout['phase_keys']['image1']:
            binding = layout['bindings'][key]
            pair = binding['category'], binding['model']
            if pair not in identities:
                identities.append(pair)
        for i, (category, model) in enumerate(identities):
            name = f'public_spare_{i+1}'
            reference = next(key for key in layout['phase_keys']['image1'] if
                (layout['bindings'][key]['category'], layout['bindings'][key]['model']) == (category, model))
            size = layout['source_objects'][reference]['bbox_size_m']
            dataset.append({'type': 'DatasetObject', 'name': name, 'category': category, 'model': model,
                'position': [spare_xy[0] + (i - (len(identities) - 1) / 2) * .19, spare_xy[1],
                             floor + .72 + size[2] / 2 + .002],
                'orientation': [0., 0., 0., 1.], 'fixed_base': False, 'visual_only': False})
            spares.append({'binding_key': name, 'binding': {'name': name, 'category': category, 'model': model},
                'public_label': f'spare {i+1} on the yellow spare table', 'public_category': category,
                'support_binding': 'change_spare_table'})
    return {'task_edit2_temporal': True, 'change_destination': {'binding_key': 'change_collection_tray',
            'public_label': 'the blue-rimmed collection and inspection tray', 'relation': 'on_top'},
        'change_apparatus': {'primitives': rows, 'dataset_objects': dataset,
            'floor_contact_names': [r['name'] for r in rows if '_leg_' in r['name']],
            'proposal_status': 'native_floor_support_collision_reach_and_fit_pending'},
        'change_spares': spares}


def build(output, seed=7):
    output = Path(output).resolve()
    src = source(HISTORY_SOURCE, 'Agent Observation')
    results = [(src, history.CategoryConverter().convert(src, ConversionContext(output, seed, history_config(src))))]
    for branch, identity in CHANGE_SOURCES.items():
        src = source(identity, 'Unobserved Change')
        result = change.CategoryConverter().convert(src, ConversionContext(output, seed, change_config(src)))
        if result.get('physical_branch') != branch:
            raise ValueError('Generated temporal branch does not match its exact source event.')
        results.append((src, result))
    manifest = {'schema': 'task_edit2_temporal_candidates_v1', 'final_selection': 'provisional',
        'native_experiments_run': False, 'supported_source_branches': list(CHANGE_SOURCES),
        'unsupported_source_branches': ['moved'], 'candidates': []}
    for src, result in results:
        task, scene = result.get('task_definition'), result.get('scene_definition')
        if task is None or scene is None:
            raise ValueError('A selected representative failed to produce concrete task and scene definitions.')
        _prepare_candidate(src, result)
        task_definition(task)
        SceneDefinition.model_validate(scene)
        signature = digest({'task': task, 'scene': scene})[:24]
        key = 'temporal_' + signature
        task['id'], scene['id'] = 'task_' + key, 'scene_' + key
        directory = output / 'candidates' / key
        directory.mkdir(parents=True, exist_ok=True)
        for definition, filename in ((task, 'task.yaml'), (scene, 'scene.yaml')):
            (directory / filename).write_text(yaml.safe_dump(definition, sort_keys=False))
        record = {k: deepcopy(v) for k, v in result.items() if k not in ('task_definition', 'scene_definition')}
        record.update(candidate_id=key, task_file=str(directory / 'task.yaml'), scene_file=str(directory / 'scene.yaml'))
        write_json(directory / 'record.json', record)
        manifest['candidates'].append(record)
    write_json(output / 'manifest.json', manifest)
    return manifest


def _prepare_candidate(src, result):
    """Attach shared placement only to the explicitly owned physical changes."""
    task, scene = result['task_definition'], result['scene_definition']
    if (result.get('source_identity') != src.relative_path or
            any(definition.get('source', {}).get('json') != src.relative_path for definition in (task, scene))):
        raise ValueError('Direct temporal conversion must preserve its exact explicit source mapping.')
    revised_history = (result.get('operation_recipe') == history.HISTORY_OPERATION and
        task['private']['evaluation']['parameters'].get('history_inventory_revision') == CORE_HISTORY_REVISION)
    if result.get('operation_recipe') != change.PHYSICAL_OPERATION and not revised_history:
        # Historical history prototypes retain their own apparatus initializer.
        # Its legacy generic policy has no explicit placement groups; this fix
        # must not activate or redesign that unrelated policy.
        result['interaction_policy_application'] = {'applied': False,
            'reason': 'historical_history_apparatus_initializer_retained_generic_policy_unapplied',
            'native_action_qualification': 'unavailable'}
        result['warning_codes'] = sorted(set(result.get('warning_codes', []) +
            ['legacy_history_generic_interaction_policy_unapplied']))
        result.setdefault('limitations', []).append('The historical history prototype uses its own apparatus '
            'initializer. Its generic interaction policy lacks explicit placement groups and is not attached '
            'by this change; native action qualification remains unavailable.')
        return
    interaction = prepare_definitions(src, scene, task, result.get('interaction_policy'))
    if interaction is not None:
        # Match ConverterBase.convert_all normalization; do not reconstruct or
        # augment the converter's scientific grouping policy here.
        result['interaction_policy'] = interaction
        result['reuse_level'] = 'L2'
        result['warning_codes'] = sorted(set(result.get('warning_codes', []) +
            list(interaction.get('warning_codes', ())) + ['interaction_layout_native_review_pending']))
        result.setdefault('change_summary', []).append(
            'Screen the actual R1 floor stance while preserving the original floorplan and furniture.'
            if interaction['mode'] == 'navigation' else
            'Screen the actual R1 floor stance and object heights; relocate only complete '
            'authorized groups onto measured existing supports without resizing.')


def _core_mapping(original, src):
    """Bind a derivative or pending decision to one exact original Core member."""
    result = {'source_identity': src.relative_path, 'source_file': str(src.path.resolve()),
        'source_sha256': sha256(src.path), 'source_core_candidate_id': original['candidate_id']}
    for kind in ('task', 'scene'):
        filename = original.get(kind + '_file')
        if filename:
            path = Path(filename).resolve()
            result['historical_' + kind + '_file'] = str(path)
            result['historical_' + kind + '_sha256'] = sha256(path) if path.is_file() else None
    return result


def _write_core_candidate(result, output, mapping, layout, src):
    task, scene = result.get('task_definition'), result.get('scene_definition')
    if task is None or scene is None:
        raise ValueError(result.get('failure_reason') or 'Exact Core source did not produce concrete definitions.')
    is_history = (result.get('operation_recipe') == history.HISTORY_OPERATION and
        task['private']['evaluation']['parameters'].get('history_inventory_revision') == CORE_HISTORY_REVISION)
    if result.get('operation_recipe') != change.PHYSICAL_OPERATION and not is_history:
        raise ValueError('Core direct writing requires the exact owned physical change or approved history operation.')
    if (mapping.get('source_identity') != src.relative_path or
            Path(mapping['source_file']).resolve() != src.path.resolve() or mapping['source_sha256'] != sha256(src.path)):
        raise ValueError('Core writing source differs from the exact explicit source mapping.')
    provenance = {key: mapping[key] for key in ('source_identity', 'source_sha256', 'source_core_candidate_id')}
    task['private']['evaluation']['parameters'].update(provenance)
    for definition in (task, scene):
        for patch in definition['scene']['variants']['adapted']['patches']:
            if patch['operation'] == 'category_initialization' and patch.get('module') in (change.MODULE, history.MODULE):
                patch['parameters'].update(provenance)
    _prepare_candidate(src, result)
    task_definition(task)
    SceneDefinition.model_validate(scene)
    key = 'temporal_core_' + digest({'task': task, 'scene': scene})[:24]
    task['id'], scene['id'] = 'task_' + key, 'scene_' + key
    directory = output / 'candidates' / key
    directory.mkdir(parents=True, exist_ok=True)
    for definition, filename in ((task, 'task.yaml'), (scene, 'scene.yaml')):
        (directory / filename).write_text(yaml.safe_dump(definition, sort_keys=False))
    record = {k: deepcopy(v) for k, v in result.items() if k not in ('task_definition', 'scene_definition')}
    record.update(candidate_id=key, task_file=str(directory / 'task.yaml'), scene_file=str(directory / 'scene.yaml'),
        task_sha256=sha256(directory / 'task.yaml'), scene_sha256=sha256(directory / 'scene.yaml'),
        source_sha256=mapping['source_sha256'], source_core_candidate_id=mapping['source_core_candidate_id'],
        core_membership='exact_original_tasks_source', final_selection_owner='user')
    if is_history:
        parameters = task['private']['evaluation']['parameters']
        record['source_inventory'] = {'original_target_keys': deepcopy(parameters['original_target_keys']),
            'original_targets': deepcopy(parameters['original_source_objects']),
            'added_target_keys': deepcopy(parameters['added_target_keys']),
            'added_targets': {key: deepcopy(parameters['source_objects'][key]) for key in parameters['added_target_keys']},
            'inventory_revision': CORE_HISTORY_REVISION, 'original_source_inventory_preserved': True}
        record['human_review_status'] = 'pending_no_native_or_model_agent_execution'
    else:
        record['source_inventory'] = {'containers': {key: deepcopy(layout['bindings'][key]) for key in layout['container_keys']},
            'phase_contents': {phase: {key: {'binding': deepcopy(layout['bindings'][key]),
                'container': layout['relations'][key]} for key in keys} for phase, keys in layout['phase_keys'].items()}}
    if result.get('physical_branch') == 'removed':
        spares = task['private']['evaluation']['parameters']['public_spare_inventory']
        record['public_spare_count'] = len(spares)
        record['spare_distractor_status'] = ('source_identity_alternatives_present' if len(spares) > 1
            else 'unavailable_only_one_distinct_source_identity')
        if len(spares) == 1:
            record['spare_inventory_limitation'] = ('This exact Core source contains one distinct phase-1 category/model. '
                'The source-supported spare table has one matching object; adding a distractor asset awaits an explicit selection.')
    write_json(directory / 'record.json', record)
    return record


def build_core(output=CORE_OUTPUT, seed=7, core_manifest=CORE_MANIFEST):
    """Generate approved Core histories and unchanged physical changes for review."""
    output = Path(output).resolve()
    if output == (ROOT / 'outputs/task_edit2_20261018/temporal').resolve():
        raise ValueError('Core mode must not overwrite the historical temporal prototype output.')
    for historical in ('temporal_core', 'temporal_core_v2'):
        if output == (ROOT / 'outputs/task_edit2_20261018' / historical).resolve():
            raise ValueError('Core mode must not overwrite the historical ' + historical + ' output; use temporal_core_v3.')
    core_manifest = Path(core_manifest).resolve()
    originals = json.loads(core_manifest.read_text())['tasks']
    selected = [row for row in originals if row['source'].startswith((
        'Temporal Understanding/Agent Observation/', 'Temporal Understanding/Unobserved Change/'))]
    if len({row['source'] for row in selected}) != len(selected):
        raise ValueError('Exact Core temporal membership contains duplicate source identities.')
    manifest = {'schema': 'task_edit2_temporal_core_candidates_v2', 'final_selection': 'provisional',
        'final_selection_owner': 'user', 'native_experiments_run': False, 'seed': seed,
        'human_review_status': 'requested_before_any_further_native_or_model_agent_execution',
        'history_inventory_revision': CORE_HISTORY_REVISION,
        'core_membership': {'manifest_file': str(core_manifest), 'manifest_sha256': sha256(core_manifest),
            'membership_field': 'tasks/source', 'source_count': len(selected)},
        'candidates': [], 'pending_records': [], 'source_mapping': [],
        'supported_source_branches': [], 'not_present_in_core_source_branches': ['moved', 'replaced']}
    for original in selected:
        identity = original['source']
        category = identity.split('/')[1]
        src = source(identity, category)
        mapping = _core_mapping(original, src)
        try:
            if category == 'Agent Observation':
                layout, branch = None, None
                config = core_history_config(src)
                result = history.CategoryConverter().convert(src, ConversionContext(output, seed, config))
            else:
                if identity not in CORE_CHANGE_BRANCHES:
                    raise ValueError('This source is outside the four explicitly reviewed Core change sources.')
                layout = source_change._layout(src)
                branch = source_change.source_change_recipe(layout)['branch']
                if branch != CORE_CHANGE_BRANCHES[identity]:
                    raise ValueError('The source event differs from the reviewed exact Core branch.')
                result = change.CategoryConverter().convert(src, ConversionContext(output, seed, change_config(src)))
                if result.get('physical_branch') != branch:
                    raise ValueError('Generated temporal branch does not match the exact Core source event.')
            record = _write_core_candidate(result, output, mapping, layout, src)
        except (ValueError, KeyError, TypeError) as error:
            pending = {**deepcopy(mapping), 'status': 'pending_source_conversion', 'generated_definition': False,
                'reason': str(error), 'native_execution_status': 'unavailable'}
            manifest['pending_records'].append(pending)
            mapping['status'] = pending['status']
        else:
            manifest['candidates'].append(record)
            mapping.update(status='generated', derivative_candidate_id=record['candidate_id'],
                task_file=record['task_file'], scene_file=record['scene_file'],
                task_sha256=record['task_sha256'], scene_sha256=record['scene_sha256'])
            if category == 'Agent Observation':
                mapping.update(history_inventory_revision=CORE_HISTORY_REVISION,
                    original_source_target_count=1, approved_added_target_count=2)
            else:
                mapping['physical_branch'] = branch
            if branch is not None and branch not in manifest['supported_source_branches']:
                manifest['supported_source_branches'].append(branch)
        manifest['source_mapping'].append(mapping)
    manifest['counts'] = {'core_temporal_sources': len(selected), 'generated_candidates': len(manifest['candidates']),
        'pending_history_inventory_redesign': sum(row['status'] == 'pending_history_inventory_redesign' for row in manifest['pending_records']),
        'pending_source_conversion': sum(row['status'] == 'pending_source_conversion' for row in manifest['pending_records'])}
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'source_mapping.json', manifest['source_mapping'])
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--core-manifest', type=Path, nargs='?', const=CORE_MANIFEST,
        help='Generate approved three-target Core histories and physical changes in temporal_core_v3 for human review.')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    result = (build_core(args.output or CORE_OUTPUT, args.seed, args.core_manifest) if args.core_manifest else
              build(args.output or ROOT / 'outputs/task_edit2_20261018/temporal', args.seed))
    print(f"Generated {len(result['candidates'])} provisional temporal task/scene pairs; native gates remain pending.")
