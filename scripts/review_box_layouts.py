"""Translate six reviewed temporal box layouts without changing their contracts.

Import ``revise(alias, task_dict, scene_dict)`` for an in-memory revision. Run
CPU validation from the repository root with the behavior environment:
  PYTHONPATH=src:../Show-Harness-Explore python scripts/review_box_layouts.py --validate
Optional ``--report PATH`` writes the validation report; release files are never
written by this script. Native support, visibility, and reachability remain
pending until checked in the simulator.
"""
import argparse
from copy import deepcopy
from itertools import combinations
import json
import math
from pathlib import Path


TRANSLATIONS_WORLD_M = {
    'T40': (0.0, -0.12, 0.0),
    'T41': (0.12, 0.0, 0.0),
    'T45': (0.12, 0.0, 0.0),
    'T46': (0.0, -0.12, 0.0),
    'T47': (0.12, 0.0, 0.0),
    'T48': (0.12, 0.0, 0.0),
}
VARIANTS = ('adapted', 'saved_image1', 'saved_image2')
MARKER = 'user_review_access_v1'
TEMPORAL_MODULES = {
    'wia.authoring.converters.temporal_understanding_two_phase',
    'wia.authoring.converters.temporal_understanding_unobserved_change',
}


def _independent_copy(value):
    """Copy each occurrence independently so YAML anchors cannot move twice."""
    if isinstance(value, dict):
        return {key: _independent_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_independent_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_independent_copy(item) for item in value)
    return deepcopy(value)


def _get(value, path):
    for key in path:
        value = value[key]
    return value


def _set(value, path, item):
    _get(value, path[:-1])[path[-1]] = item


def _source_paths(source_objects, root, affected):
    if set(source_objects) != affected:
        raise ValueError(f'Unexpected live source binding set at {root}.')
    paths = {}
    for binding, source in source_objects.items():
        paths[(*root, binding, 'position')] = binding
        if 'bbox' in source:
            if len(source['bbox']) != 2:
                raise ValueError(f'Expected two world-coordinate bbox endpoints for {binding}.')
            for endpoint in range(2):
                paths[(*root, binding, 'bbox', endpoint)] = binding
    return paths


def _live_paths(definition, affected, include_evaluation=False):
    """Enumerate only the known active coordinate interfaces, never provenance."""
    paths = {}
    if include_evaluation:
        root = ('private', 'evaluation', 'parameters', 'source_objects')
        paths.update(_source_paths(_get(definition, root), root, affected))
    variants = definition['scene']['variants']
    for variant in VARIANTS:
        patches = variants[variant]['patches']
        for index, patch in enumerate(patches):
            root = ('scene', 'variants', variant, 'patches', index)
            if patch['operation'] == 'object_pose' and patch['binding'] in affected:
                paths[(*root, 'position')] = patch['binding']
            elif patch['operation'] == 'category_initialization':
                parameters = patch['parameters']
                if 'source_objects' not in parameters:
                    continue
                if patch['module'] not in TEMPORAL_MODULES:
                    raise ValueError('An unknown initialization module owns live source coordinates.')
                paths.update(_source_paths(parameters['source_objects'],
                    (*root, 'parameters', 'source_objects'), affected))
    return paths


def _affected_bindings(task, scene):
    parameters = task['private']['evaluation']['parameters']
    containers = set(parameters['container_keys'])
    if containers != {'container_00', 'container_01', 'container_02'}:
        raise ValueError('The reviewed layouts require exactly the three original cartons.')
    phases = parameters['phase_keys']
    if set(phases) != {'image1', 'image2'}:
        raise ValueError('The reviewed layouts require both original temporal phases.')
    contents = set(phases['image1']) | set(phases['image2'])
    if any(not key.startswith(f'{phase}_content_') for phase in phases for key in phases[phase]):
        raise ValueError('Unexpected temporal source content binding.')
    affected = containers | contents
    if task['private']['bindings'] != scene['bindings']:
        raise ValueError('Task and scene bindings disagree.')
    for key in affected:
        name = scene['bindings'][key]['name']
        prefix = 'render_unobserved_box_' if key in containers else 'render_unobserved_content_'
        if not name.startswith(prefix):
            raise ValueError(f'{key} does not identify an original source carton or content.')
    return affected


def revise(alias, task_dict, scene_dict):
    """Return independent translated task/scene copies, ``adapted``, and evidence.

    Only the reviewed world XYZ translation is permitted. All source cartons and
    both phases' contents move together on the same native supports. The repair
    marker prevents repeated application. No runtime or source JSON is changed.
    """
    alias = alias.upper()
    if alias not in TRANSLATIONS_WORLD_M:
        raise ValueError(f'Unsupported reviewed box layout: {alias}.')
    if MARKER in task_dict.get('repair', {}):
        raise ValueError(f'{alias} already has the {MARKER} revision.')
    if task_dict['scene'] != scene_dict['scene']:
        raise ValueError('Task scene and standalone scene must agree before revision.')
    task, scene = _independent_copy(task_dict), _independent_copy(scene_dict)
    affected = _affected_bindings(task, scene)
    delta = TRANSLATIONS_WORLD_M[alias]
    original_source = task['private']['evaluation']['parameters']['source_objects']
    centers_before = {key: deepcopy(original_source[key]['position'])
        for key in sorted(affected) if key.startswith('container_')}
    counts = {}
    for name, definition in (('task', task), ('scene', scene)):
        paths = _live_paths(definition, affected, include_evaluation=name == 'task')
        for path in paths:
            position = _get(definition, path)
            if len(position) != 3 or not all(math.isfinite(float(value)) for value in position):
                raise ValueError(f'Expected a finite world XYZ coordinate at {path}.')
            _set(definition, path, [value + shift for value, shift in zip(position, delta)])
        counts[name] = len(paths)
        definition['version'] += 1
    if task['scene'] != scene['scene']:
        raise ValueError('Task and standalone scenes diverged during revision.')
    native_supports = sorted(set(task.get('repair', {}).get('support_name_map', {}).values()))
    task.setdefault('repair', {})[MARKER] = {
        'revision': MARKER,
        'alias': alias,
        'translation_world_m': list(delta),
        'affected_bindings': sorted(affected),
        'original_native_supports_unchanged': True,
        'native_support_names': native_supports,
        'native_validation': 'pending',
    }
    changes = {
        'alias': alias,
        'selected_variant': 'adapted',
        'translation_world_m': list(delta),
        'container_centers_before_world_m': centers_before,
        'container_centers_after_world_m': {
            key: deepcopy(task['private']['evaluation']['parameters']['source_objects'][key]['position'])
            for key in centers_before},
        'source_bindings': sorted(affected),
        'phase_bindings': deepcopy(task['private']['evaluation']['parameters']['phase_keys']),
        'updated_variants': list(VARIANTS),
        'live_coordinate_counts': counts,
        'original_native_supports_unchanged': True,
        'native_support_names': native_supports,
        'invariants': [
            'One shared world XYZ translation preserves all source-relative geometry and temporal endpoints.',
            'Native furniture, robot, apparatus, public spare inventory, and support identities are unchanged.',
            'Dimensions, scales, orientations, masses, source identity, public text, labels, timing, events, and scoring are unchanged.',
            'Raw replay payloads, historical repair metadata, and parent provenance are unchanged.',
            'Task and scene IDs are retained; each definition version is incremented once.',
        ],
        'pending_validation': ['native support/contact', 'native collision and settling',
            'visibility', 'robot and gripper reachability'],
    }
    return task, scene, 'adapted', changes


def _validate_pair(alias, task, scene):
    """Check numerical and schema invariants without importing a simulator."""
    from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition

    before_task, before_scene = _independent_copy(task), _independent_copy(scene)
    revised_task, revised_scene, variant, changes = revise(alias, task, scene)
    assert task == before_task and scene == before_scene, 'Inputs were mutated.'
    CategoryTaskDefinition.model_validate(revised_task)
    SceneDefinition.model_validate(revised_scene)
    assert variant == 'adapted' and revised_task['scene'] == revised_scene['scene']
    affected = _affected_bindings(task, scene)
    delta = TRANSLATIONS_WORLD_M[alias]
    for name, old, new in (('task', task, revised_task), ('scene', scene, revised_scene)):
        paths = _live_paths(old, affected, include_evaluation=name == 'task')
        assert paths == _live_paths(new, affected, include_evaluation=name == 'task')
        stripped = _independent_copy(new)
        for path in paths:
            original, translated = _get(old, path), _get(new, path)
            assert all(math.isclose(after - before, shift, rel_tol=0, abs_tol=1e-12)
                for before, after, shift in zip(original, translated, delta)), path
            _set(stripped, path, deepcopy(original))
        assert new['id'] == old['id'] and new['version'] == old['version'] + 1
        stripped['version'] = old['version']
        if name == 'task':
            stripped['repair'].pop(MARKER)
        assert stripped == old, f'Unexpected non-coordinate change in {name}.'
        for variant_name in VARIANTS:
            for patch in new['scene']['variants'][variant_name]['patches']:
                if patch['operation'] == 'category_initialization' and 'source_objects' in patch['parameters']:
                    assert patch['parameters']['source_objects'] == revised_task['private']['evaluation']['parameters']['source_objects']
                if patch['operation'] == 'object_pose' and patch['binding'] in affected:
                    expected = revised_task['private']['evaluation']['parameters']['source_objects'][patch['binding']]
                    assert patch['position'] == expected['position']
    old_source = task['private']['evaluation']['parameters']['source_objects']
    new_source = revised_task['private']['evaluation']['parameters']['source_objects']
    for first, second in combinations(sorted(affected), 2):
        assert all(math.isclose(new_source[first]['position'][axis] - new_source[second]['position'][axis],
            old_source[first]['position'][axis] - old_source[second]['position'][axis],
            rel_tol=0, abs_tol=1e-12) for axis in range(3))
    try:
        revise(alias, revised_task, revised_scene)
    except ValueError as error:
        assert MARKER in str(error)
    else:
        raise AssertionError('Repeated application was accepted.')
    return {**changes, 'cpu_validation': 'passed', 'schema_validation': 'passed',
        'all_non_coordinate_content_unchanged': True, 'inputs_unchanged': True,
        'pairwise_relative_geometry_preserved': True, 'both_temporal_phases_consistent': True,
        'reapplication_refused': True, 'physical_reachability_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate', action='store_true', required=True)
    parser.add_argument('--release-root', type=Path,
        default=Path('explore_to_achieve/wia_v1/release'))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    import yaml

    reports = []
    for alias in TRANSLATIONS_WORLD_M:
        task = yaml.safe_load((args.release_root / 'tasks' / f'{alias.lower()}.yaml').read_text())
        scene = yaml.safe_load((args.release_root / 'scenes' / f'{alias.lower()}.yaml').read_text())
        reports.append(_validate_pair(alias, task, scene))
    report = {'validation': 'passed', 'mode': 'CPU only; no simulator or GPU run',
        'release_root': str(args.release_root.resolve()), 'tasks': reports}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'validation': 'passed', 'aliases': list(TRANSLATIONS_WORLD_M),
        'live_coordinate_counts': {row['alias']: row['live_coordinate_counts'] for row in reports},
        'native_validation': 'pending', 'report': str(args.report) if args.report else None}, indent=2))


if __name__ == '__main__':
    main()
