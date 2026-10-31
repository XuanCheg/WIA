"""Apply the main-authored T46 access proposal to independent in-memory copies.

Import ``revise(task, scene)`` to obtain revised copies and a change record.
Run read-only CPU validation from the repository root with:
  PYTHONPATH=src /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -B \
    scripts/review_t46_access_20261031.py --check-active
No YAML, catalogue, provenance, runtime code, or native evidence is written.
Native settling, support, visibility, grasp access and physical success remain
pending. Existing saved snapshots receive content, chair and robot-start changes;
apparatus and reviewed group placements are revised only where already present.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from math import isclose, prod
from pathlib import Path


REVISION = 'review_t46_access_20261031'
PLACEMENT_KEY = 'reviewed_group_placements_v1'
SUPPORT_NAME = 'breakfast_table_zypvuv_0'
ROOT_XY_M = {'container_00': [4.05, 6.33], 'container_01': [4.65, 6.33],
             'container_02': [5.25, 6.33]}
CONTENT_RAISE_M = 0.018
BUTTER_BINDING = 'public_spare_3'
BUTTER_SCALE_RATIO = 3.0
BUTTER_MEASURED_HEIGHT_M = 0.015579
BUTTER_RAISE_M = 0.01558
RIM_HEIGHT_BEFORE_M = 0.06
RIM_HEIGHT_AFTER_M = 0.01
RIM_CENTER_DELTA_M = -0.025
RIM_NAMES = frozenset(f'change_spare_rim_{index}' for index in range(4))
NATIVE_FURNITURE_MODULE = 'wia.authoring.converters.wia_native_furniture'
CHAIR_ROOT_XY_M = {'straight_chair_eospnr_0': [6.15, 7.65],
                   'straight_chair_eospnr_5': [2.8, 5.5]}
ROBOT_TRANSLATION_WORLD_M = (0.5, 0.0, 0.0)
TEMPORAL_MODULES = frozenset({
    'wia.authoring.converters.temporal_understanding_two_phase',
    'wia.authoring.converters.temporal_understanding_unobserved_change',
})


def _copy(value):
    """Detach each occurrence so repeated YAML anchors never receive two deltas."""
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy(item) for item in value)
    return deepcopy(value)


def _placements():
    return {key: {'support_name': SUPPORT_NAME, 'root_xy_m': list(xy)}
            for key, xy in ROOT_XY_M.items()}


def _raised(position, delta):
    return [*position[:2], position[2] + delta]


def _is_butter_scale(patch):
    return patch.get('operation') == 'object_scale' and patch.get('binding') == BUTTER_BINDING


def _has_butter(apparatus):
    return any(item['name'] == BUTTER_BINDING for item in apparatus.get('dataset_objects', []))


def _revise_parameters(parameters, contents):
    counts = {'source_object_sets': 0, 'content_positions': 0, 'apparatus_sets': 0,
              'butter_positions': 0, 'spare_rims': 0}
    if 'source_objects' in parameters:
        sources = parameters['source_objects']
        if not contents <= sources.keys():
            raise ValueError('Each live source_objects mapping must retain both temporal phases.')
        for key in contents:
            sources[key]['position'] = _raised(sources[key]['position'], CONTENT_RAISE_M)
        counts.update(source_object_sets=1, content_positions=len(contents))
    if 'apparatus' in parameters:
        apparatus = parameters['apparatus']
        counts['apparatus_sets'] = 1
        for item in apparatus.get('dataset_objects', []):
            if item['name'] == BUTTER_BINDING:
                if (item['category'], item['model']) != ('butter', 'ovirmc'):
                    raise ValueError('The reviewed spare must retain its original butter asset.')
                item['position'] = _raised(item['position'], BUTTER_RAISE_M)
                counts['butter_positions'] += 1
        for item in apparatus.get('primitives', []):
            if item['name'] in RIM_NAMES:
                if not isclose(item['dimensions_m'][2], RIM_HEIGHT_BEFORE_M, abs_tol=1e-12):
                    raise ValueError('Each reviewed spare rim must start at 0.06 m high.')
                item['dimensions_m'] = [*item['dimensions_m'][:2], RIM_HEIGHT_AFTER_M]
                item['position'] = _raised(item['position'], RIM_CENTER_DELTA_M)
                counts['spare_rims'] += 1
        if counts['butter_positions'] != 1 or counts['spare_rims'] != 4:
            raise ValueError('Each existing apparatus must retain the butter spare and four spare rims.')
    return counts


def _revise_scene(config, contents):
    scopes = {}
    for name, variant in config['variants'].items():
        if name == 'original':
            if variant['patches']:
                raise ValueError('The schema-required original variant must remain empty.')
            continue
        counts = {'source_object_sets': 0, 'content_positions': 0, 'apparatus_sets': 0,
                  'butter_positions': 0, 'spare_rims': 0, 'content_pose_patches': 0,
                  'spare_rim_pose_patches': 0, 'butter_pose_patches': 0,
                  'reviewed_group_patches': 0, 'butter_scale_patches': 0,
                  'native_furniture_pose_patches': 0, 'chair_positions': 0, 'robot_pose_patches': 0}
        patches = variant['patches']
        for patch in patches:
            operation = patch['operation']
            if operation == 'category_initialization':
                parameters = patch['parameters']
                if 'source_objects' in parameters or 'apparatus' in parameters:
                    if patch['module'] not in TEMPORAL_MODULES:
                        raise ValueError('An unknown initializer owns the reviewed live geometry.')
                    for key, value in _revise_parameters(parameters, contents).items():
                        counts[key] += value
                if patch['module'] == NATIVE_FURNITURE_MODULE:
                    counts['native_furniture_pose_patches'] += 1
                    for item in parameters['native_object_poses']:
                        if item['name'] in CHAIR_ROOT_XY_M:
                            if (item['category'], item['model']) != ('straight_chair', 'eospnr'):
                                raise ValueError('Reviewed chairs must retain their original native assets.')
                            item['position'] = [*CHAIR_ROOT_XY_M[item['name']], item['position'][2]]
                            counts['chair_positions'] += 1
            elif operation == 'robot_pose':
                patch['position'] = [value + delta for value, delta in
                                     zip(patch['position'], ROBOT_TRANSLATION_WORLD_M)]
                counts['robot_pose_patches'] += 1
            elif operation == 'object_pose':
                binding = patch['binding']
                if binding in contents:
                    patch['position'] = _raised(patch['position'], CONTENT_RAISE_M)
                    counts['content_pose_patches'] += 1
                elif binding in RIM_NAMES:
                    patch['position'] = _raised(patch['position'], RIM_CENTER_DELTA_M)
                    counts['spare_rim_pose_patches'] += 1
                elif binding == BUTTER_BINDING:
                    patch['position'] = _raised(patch['position'], BUTTER_RAISE_M)
                    counts['butter_pose_patches'] += 1
            elif operation == 'prepare_category_interaction':
                patch['relocation_policy'] = 'independent_groups'
                patch.setdefault('layout_parameters', {})[PLACEMENT_KEY] = _placements()
                counts['reviewed_group_patches'] += 1
        creators = [patch for patch in patches if patch['operation'] == 'category_initialization'
                    and patch.get('phase', 'before_construction') == 'before_construction'
                    and _has_butter(patch['parameters'].get('apparatus', {}))]
        scales = [patch for patch in patches if _is_butter_scale(patch)]
        if creators:
            if len(creators) != 1 or any(patch.get('phase', 'before_construction') != 'before_construction'
                                         for patch in scales):
                raise ValueError('Butter requires exactly one preconstruction apparatus creator and scale phase.')
            multiplier = prod(patch['multiplier'] for patch in scales) * BUTTER_SCALE_RATIO
            patches = [patch for patch in patches if not _is_butter_scale(patch)]
            creator_index = next(index for index, patch in enumerate(patches) if patch is creators[0])
            patches.insert(creator_index + 1, {'operation': 'object_scale', 'phase': 'before_construction',
                                              'binding': BUTTER_BINDING, 'multiplier': multiplier})
            variant['patches'] = patches
            counts.update(butter_scale_patches=1, butter_effective_multiplier=multiplier,
                          previous_butter_scale_patch_count=len(scales))
        elif scales:
            raise ValueError('Butter scale patches require an existing apparatus in the same variant.')
        scopes[name] = counts
    if not scopes.get('adapted', {}).get('reviewed_group_patches'):
        raise ValueError('The selected adapted variant must have its existing interaction setup.')
    return scopes


def revise(task: dict, scene: dict) -> tuple[dict, dict, dict]:
    """Return independent revised copies and the exact main-authored geometry scope."""
    if REVISION in task.get('repair', {}):
        raise ValueError(f'{REVISION} has already been applied.')
    if 'repair' in scene:
        raise ValueError('Standalone SceneDefinition forbids a repair field.')
    if task['scene'] != scene['scene'] or task['private']['bindings'] != scene['bindings']:
        raise ValueError('Task and standalone scene initialization and bindings must match.')
    evaluation = task['private']['evaluation']
    parameters = evaluation['parameters']
    if (task.get('repair', {}).get('document_alias') != 'T46'
            or evaluation.get('module') != 'wia.authoring.converters.temporal_understanding_two_phase'
            or evaluation.get('operation') != 'compare_change_and_restore_or_collect'
            or set(parameters['container_keys']) != set(ROOT_XY_M)):
        raise ValueError('This helper requires the current T46 three-carton temporal contract.')
    contents = set().union(*parameters['phase_keys'].values())
    if not contents or any(not key.startswith(('image1_content_', 'image2_content_')) for key in contents):
        raise ValueError('The reviewed source contents must retain both original temporal phases.')
    revised_task, revised_scene = _copy(task), _copy(scene)
    evaluation_scope = _revise_parameters(revised_task['private']['evaluation']['parameters'], contents)
    task_scope = _revise_scene(revised_task['scene'], contents)
    scene_scope = _revise_scene(revised_scene['scene'], contents)
    if revised_task['scene'] != revised_scene['scene'] or task_scope != scene_scope:
        raise ValueError('Task and scene live geometry diverged during revision.')
    revised_task['version'] += 1
    revised_scene['version'] += 1
    geometry = {'reviewed_group_placements_v1': _placements(),
                'relocation_policy': 'independent_groups', 'content_center_raise_m': CONTENT_RAISE_M,
                'butter_scale_ratio': BUTTER_SCALE_RATIO, 'butter_center_raise_m': BUTTER_RAISE_M,
                'butter_measured_height_before_m': BUTTER_MEASURED_HEIGHT_M,
                'butter_predicted_height_after_m': BUTTER_MEASURED_HEIGHT_M * BUTTER_SCALE_RATIO,
                'spare_rim_height_before_m': RIM_HEIGHT_BEFORE_M,
                'spare_rim_height_after_m': RIM_HEIGHT_AFTER_M,
                'spare_rim_center_delta_m': RIM_CENTER_DELTA_M,
                'chair_root_xy_m': _copy(CHAIR_ROOT_XY_M),
                'robot_translation_world_m': list(ROBOT_TRANSLATION_WORLD_M)}
    revised_task.setdefault('repair', {})[REVISION] = {
        'revision': REVISION, 'alias': 'T46', **_copy(geometry),
        'basis': 'Main-authored proposal from native evidence: content bottoms were about 0.008 m below '
                 'carton bounds; a 0.018 m raise proposes about 0.010 m settling clearance. The butter '
                 'measured height was 0.015579 m; uniform 3x scale adds 0.015579 m to its half-height, '
                 'rounded upward to a 0.01558 m center raise. Spare rims retain their original bottom '
                 'Z while lowering from 0.06 m to 0.01 m for access. Cartons retain their assets and '
                 'authored source poses; only existing interaction setup receives exact native-table XY targets. '
                 'Use root Y=6.33 m: the main agent measured the open-carton minimum Y as 6.217809 m '
                 'at root Y=6.48 m and the table minimum Y as 6.015751 m. The corrected target predicts '
                 'carton minimum Y=6.067809 m, retaining the existing 0.03 m support inset; the earlier '
                 'Y=6.25 m proposal would overhang and is discarded. '
                 'Two front chairs blocked useful front-table stances in the main agent\'s bounded CPU audit. '
                 'Move only their existing native-furniture XY poses and shift configured robot starts +0.5 m '
                 'world X. The main agent reports no mesh-AABB overlaps above 0.005 m for the proposed chairs '
                 'and a clear predicted robot SE2 footprint; these prospective screens do not verify native '
                 'settling, grasp access, or physical task success.',
        'variant_scope': _copy(task_scope), 'native_validation': 'pending',
        'native_validation_run': False, 'physical_success_verified': False,
    }
    return revised_task, revised_scene, {'revision': REVISION, 'selected_variant': 'adapted',
        'geometry': geometry, 'evaluation_scope': evaluation_scope, 'per_variant_scope': task_scope,
        'content_bindings': sorted(contents), 'native_validation': 'pending', 'physical_success_verified': False}


def _audit_parameters(old, new, contents, canonical):
    restored = _copy(new)
    if 'source_objects' in old:
        assert new['source_objects'] == canonical['source_objects'], 'Live source copies disagree.'
        for binding, source in old['source_objects'].items():
            expected = _raised(source['position'], CONTENT_RAISE_M) if binding in contents else source['position']
            assert new['source_objects'][binding]['position'] == expected
            restored['source_objects'][binding]['position'] = _copy(source['position'])
    if 'apparatus' in old:
        assert new['apparatus'] == canonical['apparatus'], 'Live apparatus copies disagree.'
        for field in ('dataset_objects', 'primitives'):
            for before, after, reset in zip(old['apparatus'].get(field, []), new['apparatus'].get(field, []),
                                            restored['apparatus'].get(field, [])):
                if before['name'] == BUTTER_BINDING:
                    assert after['position'] == _raised(before['position'], BUTTER_RAISE_M)
                    reset['position'] = _copy(before['position'])
                elif before['name'] in RIM_NAMES:
                    assert after['position'] == _raised(before['position'], RIM_CENTER_DELTA_M)
                    assert after['dimensions_m'] == [*before['dimensions_m'][:2], RIM_HEIGHT_AFTER_M]
                    assert isclose(after['position'][2] - after['dimensions_m'][2] / 2,
                                   before['position'][2] - before['dimensions_m'][2] / 2, abs_tol=1e-12)
                    reset['position'], reset['dimensions_m'] = _copy(before['position']), _copy(before['dimensions_m'])
    if 'native_object_poses' in old:
        assert len(new['native_object_poses']) == len(old['native_object_poses'])
        for before, after, reset in zip(old['native_object_poses'], new['native_object_poses'],
                                        restored['native_object_poses']):
            if before['name'] in CHAIR_ROOT_XY_M:
                assert after['position'] == [*CHAIR_ROOT_XY_M[before['name']], before['position'][2]]
                reset['position'] = _copy(before['position'])
    assert restored == old, 'A parameter outside the reviewed live geometry changed.'


def _audit_pair(task, scene):
    """Check exact change masks, schemas, phase consistency and duplicate refusal."""
    from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition

    originals = _copy((task, scene))
    CategoryTaskDefinition.model_validate(task)
    SceneDefinition.model_validate(scene)
    new_task, new_scene, changes = revise(task, scene)
    CategoryTaskDefinition.model_validate(new_task)
    SceneDefinition.model_validate(new_scene)
    assert (task, scene) == originals, 'Inputs changed.'
    assert new_task['scene'] == new_scene['scene']
    assert 'repair' not in new_scene
    canonical = new_task['private']['evaluation']['parameters']
    contents = set(changes['content_bindings'])
    _audit_parameters(task['private']['evaluation']['parameters'], canonical, contents, canonical)
    for old, new in ((task, new_task), (scene, new_scene)):
        restored = _copy(new)
        restored['version'], restored['scene'] = old['version'], _copy(old['scene'])
        if old is task:
            restored['repair'].pop(REVISION)
            restored['private']['evaluation']['parameters'] = _copy(old['private']['evaluation']['parameters'])
        assert restored == old, 'Source, IDs, bindings, public text, event, scorer or provenance changed.'
        assert new['version'] == old['version'] + 1
        for variant_name, variant in new['scene']['variants'].items():
            old_variant = old['scene']['variants'][variant_name]
            if variant_name == 'original':
                assert variant == old_variant and variant['patches'] == []
                continue
            before = [patch for patch in old_variant['patches'] if not _is_butter_scale(patch)]
            after = [patch for patch in variant['patches'] if not _is_butter_scale(patch)]
            assert len(before) == len(after)
            for original_patch, revised_patch in zip(before, after):
                reset = _copy(revised_patch)
                operation = original_patch['operation']
                if operation == 'category_initialization':
                    _audit_parameters(original_patch['parameters'], revised_patch['parameters'], contents, canonical)
                    reset['parameters'] = _copy(original_patch['parameters'])
                elif operation == 'robot_pose':
                    assert revised_patch['position'] == [value + delta for value, delta in
                        zip(original_patch['position'], ROBOT_TRANSLATION_WORLD_M)]
                    reset['position'] = _copy(original_patch['position'])
                elif operation == 'object_pose':
                    binding = original_patch['binding']
                    delta = (CONTENT_RAISE_M if binding in contents else RIM_CENTER_DELTA_M
                             if binding in RIM_NAMES else BUTTER_RAISE_M if binding == BUTTER_BINDING else 0)
                    if delta:
                        assert revised_patch['position'] == _raised(original_patch['position'], delta)
                        reset['position'] = _copy(original_patch['position'])
                    if binding in contents:
                        assert revised_patch['position'] == canonical['source_objects'][binding]['position']
                    elif binding in RIM_NAMES:
                        item = next(row for row in canonical['apparatus']['primitives'] if row['name'] == binding)
                        assert revised_patch['position'] == item['position']
                elif operation == 'prepare_category_interaction':
                    assert revised_patch['relocation_policy'] == 'independent_groups'
                    assert revised_patch['layout_parameters'][PLACEMENT_KEY] == _placements()
                    reset['relocation_policy'] = original_patch['relocation_policy']
                    reset['layout_parameters'] = _copy(original_patch['layout_parameters'])
                assert reset == original_patch, 'An unrelated scene patch field changed.'
            scales = [patch for patch in variant['patches'] if _is_butter_scale(patch)]
            if changes['per_variant_scope'][variant_name]['butter_scale_patches']:
                assert len(scales) == 1
                original_scales = [patch['multiplier'] for patch in old_variant['patches'] if _is_butter_scale(patch)]
                assert scales[0]['multiplier'] == prod(original_scales) * BUTTER_SCALE_RATIO
                scale_index = next(index for index, patch in enumerate(variant['patches']) if _is_butter_scale(patch))
                creator = variant['patches'][scale_index - 1]
                assert creator['operation'] == 'category_initialization'
                assert creator['phase'] == scales[0]['phase'] == 'before_construction'
                assert _has_butter(creator['parameters']['apparatus'])
            else:
                assert scales == []
    try:
        revise(new_task, new_scene)
    except ValueError as error:
        assert 'already been applied' in str(error)
    else:
        raise AssertionError('Repeated revision was accepted.')
    return changes


def _check_active(catalogue_path):
    import yaml
    from wia.tasks.loader import UniqueLoader

    root = Path(__file__).resolve().parents[1]
    entries = [row for row in json.loads(catalogue_path.read_text())['tasks'] if row['alias'] == 'T46']
    if len(entries) != 1:
        raise ValueError('The active catalogue must contain T46 exactly once.')
    paths = [root / entries[0][key] for key in ('task_file', 'scene_file')]
    original_bytes = [path.read_bytes() for path in paths]
    task, scene = [yaml.load(content, Loader=UniqueLoader) for content in original_bytes]
    changes = _audit_pair(task, scene)
    anchored_task, anchored_scene = _copy(task), _copy(scene)
    shared = anchored_task['private']['evaluation']['parameters']
    anchored_scene['scene'] = anchored_task['scene']
    for variant in anchored_task['scene']['variants'].values():
        for patch in variant['patches']:
            if patch['operation'] == 'category_initialization':
                for key in ('source_objects', 'apparatus'):
                    if key in patch['parameters']:
                        patch['parameters'][key] = shared[key]
    _audit_pair(anchored_task, anchored_scene)
    scale_task, scale_scene = _copy(task), _copy(scene)
    for definition in (scale_task, scale_scene):
        definition['scene']['variants']['adapted']['patches'].extend([
            {'operation': 'object_scale', 'phase': 'before_construction', 'binding': BUTTER_BINDING, 'multiplier': 2.0},
            {'operation': 'object_scale', 'phase': 'before_construction', 'binding': BUTTER_BINDING, 'multiplier': 0.75}])
    scale_changes = _audit_pair(scale_task, scale_scene)
    assert scale_changes['per_variant_scope']['adapted']['butter_effective_multiplier'] == 4.5
    assert [path.read_bytes() for path in paths] == original_bytes
    return {'validation': 'passed', 'mode': 'CPU-only in-memory validation',
        'task_file': str(paths[0]), 'scene_file': str(paths[1]), 'changes': changes,
        'input_files_unchanged': True, 'all_live_geometry_copies_consistent': True,
        'source_ids_bindings_public_event_scorer_and_provenance_preserved': True,
        'original_variant_empty': True, 'single_application_enforced': True,
        'yaml_anchor_occurrences_independent': True, 'existing_scale_composition_checked': True,
        'native_validation_run': False, 'physical_success_verified': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-active', action='store_true', required=True)
    parser.add_argument('--catalogue', type=Path, default=Path(__file__).resolve().parents[1]
                        / 'explore_to_achieve/wia_v1/release/catalogue.json')
    args = parser.parse_args(argv)
    print(json.dumps(_check_active(args.catalogue), indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
