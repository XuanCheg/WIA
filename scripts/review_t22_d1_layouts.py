"""Apply the main-authored T22/D1 review proposal to in-memory definitions.

Import ``revise(alias, task, scene)`` to obtain revised copies and a change record.
Run the CPU-only active-release check from the repository root with:
  PYTHONPATH=src:../Show-Harness-Explore \
    /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -B \
    scripts/review_t22_d1_layouts.py --check-active \
    --report outputs/review_repairs_20261030/t22_d1_helper_validation.json
The check reads active YAMLs without changing them. No simulator is started.
The reviewed D1 X offsets from the table origin are -0.22 m for the frame,
+0.14 m for the pad, -0.50 m for candidate_0 and -0.48 m for candidate_1.
Native geometry, robot access and physical success remain pending review.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path


REVISION = 'user_review_access_v1'
SUPPORTED_ALIASES = ('T22', 'D1-1', 'D1-2', 'D1-3')
INITIALIZER = 'wia.authoring.converters.active_layout'
D1_X_OFFSETS_M = {
    'wia_dim_aperture_low': -.22,
    'wia_dim_aperture_high': -.22,
    'wia_dim_aperture_roof': -.22,
    'wia_dim_pad_0': .14,
    'wia_dim_candidate_0': -.50,
    'wia_dim_candidate_1': -.48,
}
D1_ACCESS_CONSTRAINT = (
    'Robot base translation and rotation are allowed, including lateral movement '
    'around the workstation. The selected block must actually pass through the '
    'framed opening; the robot hand itself need not pass through the opening.'
)


def _active_layouts(definition: dict):
    """Yield only selected live initializer layouts, excluding repair provenance."""
    patches = definition['scene']['variants']['adapted']['patches']
    for index, patch in enumerate(patches):
        if (patch.get('operation') == 'category_initialization'
                and patch.get('module') == INITIALIZER):
            layout = patch.get('parameters', {}).get('layout')
            if not isinstance(layout, dict):
                raise ValueError('Every live active_layout initializer requires a layout.')
            yield index, patch, layout


def _revise_d1_layouts(definition: dict, definition_name: str) -> list[dict]:
    # Detach the active variant from aliases elsewhere in the copied definition.
    # In particular, historical repair records and evaluator data stay unchanged.
    scene_config = deepcopy(definition['scene'])
    variant = deepcopy(scene_config['variants']['adapted'])
    variant['patches'] = [deepcopy(patch) for patch in variant['patches']]
    scene_config['variants']['adapted'] = variant
    definition['scene'] = scene_config
    changes = []
    for index, patch, layout in _active_layouts(definition):
        objects = {obj['name']: obj for obj in layout['objects']}
        if len(objects) != len(layout['objects']):
            raise ValueError('Live layout object names must be unique.')
        required = {'wia_dim_table', *D1_X_OFFSETS_M}
        if not required <= objects.keys():
            raise ValueError('D1 requires the reviewed named table, frame, pad and candidates.')
        table_x = objects['wia_dim_table']['position'][0]
        positions = {}
        for name, offset in D1_X_OFFSETS_M.items():
            obj = objects[name]
            before = deepcopy(obj['position'])
            # Absolute coordinates avoid repeated deltas through YAML aliases.
            obj['position'] = [table_x + offset, *before[1:]]
            positions[name] = {'before_m': before, 'after_m': deepcopy(obj['position'])}
        changes.append({
            'definition': definition_name,
            'path': f'scene.variants.adapted.patches[{index}].parameters.layout',
            'phase': patch.get('phase'),
            'table_origin_x_m': table_x,
            'positions': positions,
        })
    if not changes:
        raise ValueError('D1 requires nonempty live adapted active_layout initialization.')
    return changes


def revise(alias: str, task: dict, scene: dict) -> tuple[dict, dict, str, dict]:
    """Return revised copies, selected variant and an auditable change dictionary.

    The fixed D1 offsets are a proposal supplied by the main reviewer. This pure
    function makes no claim that the robot can grasp or complete the task.
    """
    if alias not in SUPPORTED_ALIASES:
        raise ValueError(f'Unsupported review alias: {alias!r}.')
    if any(REVISION in value.get('repair', {}) for value in (task, scene)):
        raise ValueError(f'{REVISION} has already been applied.')
    if task['scene'] != scene['scene']:
        raise ValueError('Task and scene initialization must match before revision.')
    if task['private']['bindings'] != scene['bindings']:
        raise ValueError('Task private bindings and scene bindings must match.')
    revised_task, revised_scene = deepcopy(task), deepcopy(scene)
    selected_variant = 'tabletop' if alias == 'T22' else 'adapted'
    changes = {
        'revision': REVISION,
        'alias': alias,
        'selected_variant': selected_variant,
        'versions': {
            'task': {'before': task['version'], 'after': task['version'] + 1},
            'scene': {'before': scene['version'], 'after': scene['version'] + 1},
        },
        'coordinates': [],
        'public_constraints_added': [],
        'native_validation_status': 'pending',
        'physical_success_verified': False,
    }
    if alias == 'T22':
        patches = scene['scene']['variants'].get('tabletop', {}).get('patches')
        if not patches:
            raise ValueError('T22 requires nonempty authored tabletop patches.')
        reason = (
            'The release catalogue selected original, which has no initialization '
            'patches, instead of the authored tabletop chair/table initialization. '
            'Select tabletop and preserve every authored pose, support, scale, '
            'evaluation predicate and the final_state_v1 sequential-insertion contract. '
            'Native initialization and robot access validation are pending.'
        )
        changes['variant_selection'] = {'before': 'original', 'after': 'tabletop'}
        changes['selected_patch_count'] = len(patches)
    else:
        if task['private']['evaluation'].get('operation') != 'select_pass_aperture':
            raise ValueError('D1 review requires the select_pass_aperture contract.')
        changes['table_relative_x_offsets_m'] = deepcopy(D1_X_OFFSETS_M)
        for name, definition in (('task', revised_task), ('scene', revised_scene)):
            changes['coordinates'].extend(_revise_d1_layouts(definition, name))
        constraints = list(revised_task['public'].get('constraints', []))
        if D1_ACCESS_CONSTRAINT not in constraints:
            constraints.append(D1_ACCESS_CONSTRAINT)
            changes['public_constraints_added'].append(D1_ACCESS_CONSTRAINT)
        revised_task['public']['constraints'] = constraints
        reason = (
            'Apply the main-authored workstation access proposal: move only frame, '
            'blue pad and candidate X positions to the reviewed table-relative '
            'offsets (-0.22 m frame, +0.14 m pad, -0.50 m candidate_0, '
            '-0.48 m candidate_1), and explicitly allow lateral robot base motion. Preserve '
            'table, legs, robot start, all dimensions, orientations, Y/Z coordinates, '
            'historical provenance and final_state_v1 physical-passage predicates. '
            'Native geometry, robot access and physical success validation are pending.'
        )
    revised_task['version'] += 1
    revised_scene['version'] += 1
    revised_task.setdefault('repair', {})[REVISION] = {
        'alias': alias,
        'selected_variant': selected_variant,
        'reason': reason,
        'native_validation_status': 'pending',
        'native_validation_run': False,
        'physical_success_verified': False,
    }
    if alias != 'T22':
        revised_task['repair'][REVISION]['table_relative_x_offsets_m'] = deepcopy(D1_X_OFFSETS_M)
    if revised_task['scene'] != revised_scene['scene']:
        raise ValueError('Revised task and scene initialization must match.')
    return revised_task, revised_scene, selected_variant, changes


def _geometry_check(task: dict) -> dict:
    """Check exact reviewed cube footprints and strict near-side gate separation."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    from wia.authoring.commands.build_wia_control_family_repairs import cube_dimensions, footprint

    records = []
    for index, _, layout in _active_layouts(task):
        objects = {obj['name']: obj for obj in layout['objects']}
        table = objects['wia_dim_table']
        center = np.asarray(table['position'])
        rotation = Rotation.from_quat(table['orientation']).as_matrix()
        table_dimensions = cube_dimensions(table)
        bounds = {name: footprint(obj, center, rotation) for name, obj in objects.items()}
        candidates = {}
        for name in ('wia_dim_candidate_0', 'wia_dim_candidate_1'):
            low, high = bounds[name]
            margins = np.array([low[0] + table_dimensions[0] / 2,
                table_dimensions[0] / 2 - high[0],
                low[1] + table_dimensions[1] / 2,
                table_dimensions[1] / 2 - high[1]])
            if margins.min() < -1e-9:
                raise ValueError(f'{name} footprint is outside the table.')
            gaps = {gate: float(bounds[gate][0][0] - high[0])
                for gate in ('wia_dim_aperture_low', 'wia_dim_aperture_high', 'wia_dim_aperture_roof')}
            if min(gaps.values()) <= 1e-9:
                raise ValueError(f'{name} is not fully clear of the near gate face.')
            candidates[name] = {
                'bounds_table_frame_m': [low.tolist(), high.tolist()],
                'table_edge_margins_front_far_left_right_m': margins.tolist(),
                'near_gate_clearances_m': gaps,
            }
        records.append({'patch_index': index,
            'gate_depths_m': {name: float(cube_dimensions(objects[name])[0])
                for name in ('wia_dim_aperture_low', 'wia_dim_aperture_high', 'wia_dim_aperture_roof')},
            'candidates': candidates})
    return {'live_layouts': records, 'native_geometry_verified': False}


def _check_active(catalogue_path: Path) -> dict:
    """Validate current release definitions in memory without writing any YAML."""
    import yaml

    from wia.tasks.loader import UniqueLoader
    from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition

    root = Path(__file__).resolve().parents[1]
    catalogue = json.loads(catalogue_path.read_text())
    entries = {row['alias']: row for row in catalogue['tasks'] if row['alias'] in SUPPORTED_ALIASES}
    if entries.keys() != set(SUPPORTED_ALIASES):
        raise ValueError('Active catalogue must contain each supported alias exactly once.')
    results = []
    for alias in SUPPORTED_ALIASES:
        row = entries[alias]
        paths = [root / row[key] for key in ('task_file', 'scene_file')]
        task, scene = [yaml.load(path.read_text(), Loader=UniqueLoader) for path in paths]
        parent_task, parent_scene = deepcopy(task), deepcopy(scene)
        revised_task, revised_scene, variant, changes = revise(alias, task, scene)
        CategoryTaskDefinition.model_validate(revised_task)
        SceneDefinition.model_validate(revised_scene)
        assert task == parent_task and scene == parent_scene, 'Input dictionaries changed.'
        assert revised_task['scene'] == revised_scene['scene']
        assert revised_task['private']['bindings'] == revised_scene['bindings']
        assert revised_task['scene']['variants'][variant]['patches']
        assert revised_task['source'] == task['source']
        assert revised_scene['source'] == scene['source']
        assert revised_task['private'] == task['private'], 'Evaluator or private bindings changed.'
        assert revised_task['id'] == task['id'] and revised_scene['id'] == scene['id']
        assert revised_task['version'] == task['version'] + 1
        assert revised_scene['version'] == scene['version'] + 1
        for old, new in ((task, revised_task), (scene, revised_scene)):
            old_fixed, new_fixed = deepcopy(old), deepcopy(new)
            for key in ('version', 'scene'):
                old_fixed.pop(key)
                new_fixed.pop(key)
            if old is task:
                new_fixed['repair'].pop(REVISION)
                if alias != 'T22':
                    new_fixed['public']['constraints'] = deepcopy(old_fixed['public']['constraints'])
            assert old_fixed == new_fixed, 'A field outside the proposed revision changed.'
        if alias == 'T22':
            assert revised_task['scene'] == task['scene']
            assert revised_task['public'] == task['public']
            geometry = {'authored_initialization_unchanged': True,
                'tabletop_patch_count': len(revised_scene['scene']['variants']['tabletop']['patches'])}
        else:
            for old, new in ((task, revised_task), (scene, revised_scene)):
                before_layouts, after_layouts = list(_active_layouts(old)), list(_active_layouts(new))
                assert len(before_layouts) == len(after_layouts)
                for (_, _, before), (_, _, after) in zip(before_layouts, after_layouts):
                    restored = deepcopy(after)
                    old_objects = {obj['name']: obj for obj in before['objects']}
                    for obj in restored['objects']:
                        if obj['name'] in D1_X_OFFSETS_M:
                            assert obj['position'][1:] == old_objects[obj['name']]['position'][1:]
                            obj['position'] = deepcopy(old_objects[obj['name']]['position'])
                    assert restored == before, 'Dimensions, support, robot, or other layout data changed.'
            geometry = _geometry_check(revised_task)
        try:
            revise(alias, revised_task, revised_scene)
        except ValueError as error:
            assert 'already been applied' in str(error)
        else:
            raise AssertionError('Repeated application was not refused.')
        results.append({'alias': alias, 'task_file': str(paths[0]), 'scene_file': str(paths[1]),
            'selected_variant': variant, 'schema_validation': 'passed',
            'input_dictionaries_unchanged': True, 'source_dictionaries_unchanged': True,
            'private_bindings_and_evaluator_unchanged': True,
            'dimensions_support_robot_and_other_layout_fields_unchanged': True,
            'repeated_application_refused': True, 'changes': changes, 'geometry': geometry})
    return {'revision': REVISION, 'validation': 'passed',
        'mode': 'CPU-only in-memory active-release validation',
        'catalogue_file': str(catalogue_path.resolve()),
        'native_validation_run': False, 'physical_success_verified': False, 'entries': results}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-active', action='store_true', required=True)
    parser.add_argument('--catalogue', type=Path,
        default=Path(__file__).resolve().parents[1] / 'explore_to_achieve/wia_v1/release/catalogue.json')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args(argv)
    report = _check_active(args.catalogue)
    rendered = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered)
        print(f'CPU validation passed for {len(report["entries"])} pairs; report: {args.report}')
    else:
        print(rendered, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
