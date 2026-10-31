"""Apply the main-authored T22 gift scale revision to in-memory definitions.

Import ``revise(task, scene)`` to obtain revised copies and a change record.
Run the read-only CPU validation from the repository root with:
  PYTHONPATH=src /local_home2/zhouxinyang/miniconda3/envs/behavior/bin/python -B \
    scripts/review_t22_scale_20261031.py --check-active
No YAML is written and no simulator is started. Native support, grasp access,
and physical success with the smaller gift remain pending validation.
The schema-required empty ``original`` variant is retained; revised semantics
apply to the selected ``tabletop`` variant and every other non-original variant.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from math import prod
from pathlib import Path


REVISION = 'review_gift_scale_20261031'
SCALE_RATIO = 0.40
NATIVE_DIMENSIONS_BEFORE_M = (0.244474411, 0.186240673, 0.286185741)
INITIALIZER = 'wia.authoring.converters.active_layout'


def _is_target_scale(patch: dict) -> bool:
    return patch.get('operation') == 'object_scale' and patch.get('binding') == 'target'


def _scaled_scene(config: dict, target_name: str, support_name: str) -> tuple[dict, list[dict]]:
    """Copy every variant independently before replacing its target scale patches."""
    revised = deepcopy(config)
    revised['variants'] = {name: deepcopy(variant) for name, variant in config['variants'].items()}
    changes = []
    aligned_layouts = 0
    for name, variant in revised['variants'].items():
        if name == 'original':
            if variant['patches']:
                raise ValueError('The schema-native original variant must remain empty.')
            continue
        patches = [deepcopy(patch) for patch in variant['patches']]
        target_indices = [index for index, patch in enumerate(patches) if _is_target_scale(patch)]
        multipliers = []
        for index in target_indices:
            patch = patches[index]
            if patch.get('phase', 'before_construction') != 'before_construction':
                raise ValueError('Target scaling must occur before native construction.')
            multipliers.append(patch['multiplier'])
        effective_before = prod(multipliers)
        target_patch = {
            'operation': 'object_scale',
            'phase': 'before_construction',
            'binding': 'target',
            'multiplier': effective_before * SCALE_RATIO,
        }
        insertion_index = target_indices[0] if target_indices else 0
        variant['patches'] = [patch for patch in patches if not _is_target_scale(patch)]
        variant['patches'].insert(insertion_index, target_patch)
        for patch in patches:
            if (patch.get('operation') != 'category_initialization'
                    or patch.get('module') != INITIALIZER):
                continue
            layout = patch['parameters']['layout']
            for item in layout.get('supported_items', []):
                if item.get('name') != target_name:
                    continue
                if (patch.get('phase') != 'after_postprocessing'
                        or item.get('support') != support_name
                        or item.get('support_alignment_mode') != 'local_raycast'
                        or item.get('support_surface_mode') != 'local_raycast'
                        or not layout.get('synchronize_authored_poses_before_alignment')):
                    raise ValueError('The existing target local-raycast support reseating must be preserved.')
                aligned_layouts += 1
        changes.append({
            'variant': name,
            'previous_target_scale_patch_count': len(target_indices),
            'previous_target_multipliers': multipliers,
            'effective_multiplier_before': effective_before,
            'effective_multiplier_after': target_patch['multiplier'],
            'scale_ratio': SCALE_RATIO,
        })
    if not aligned_layouts:
        raise ValueError('T22 requires its existing target local-raycast support alignment.')
    return revised, changes


def revise(task: dict, scene: dict) -> tuple[dict, dict, dict]:
    """Return copies with only the reviewed gift scale and private answer changed."""
    if any(REVISION in definition.get('repair', {}) for definition in (task, scene)):
        raise ValueError(f'{REVISION} has already been applied.')
    if task['scene'] != scene['scene']:
        raise ValueError('Task and scene initialization must match before revision.')
    if task['private']['bindings'] != scene['bindings']:
        raise ValueError('Task private bindings and scene bindings must match.')
    bindings = task['private']['bindings']
    if bindings['target'] != {'name': 'obj_containee', 'category': 'gift_box', 'model': 'pnprrx'}:
        raise ValueError('This revision requires the reviewed T22 gift asset.')
    evaluation = task['private']['evaluation']
    parameters = evaluation['parameters']
    if (evaluation.get('operation') != 'pick_release_partial_insert'
            or parameters.get('scorer_version') != 'rigid_insertion_final_state_v1'
            or parameters.get('manipulation_revision') != 'final_state_v1'
            or parameters['expected_containers']['target'] != 'container_1'
            or parameters['expected_answers']['target'] != 'medium packing box'):
        raise ValueError('This revision requires the existing T22 final-state contract and medium-box answer.')
    revised_task, revised_scene = deepcopy(task), deepcopy(scene)
    revised_config, scale_changes = _scaled_scene(
        task['scene'], bindings['target']['name'], bindings['pickup_support']['name'])
    revised_task['scene'] = revised_config
    revised_scene['scene'] = deepcopy(revised_config)
    revised_parameters = revised_task['private']['evaluation']['parameters']
    revised_parameters['expected_containers']['target'] = 'container_0'
    revised_parameters['expected_answers']['target'] = 'small packing box'
    evidence = {
        'basis': 'Main-authored revision following the user request to uniformly shrink the gift box.',
        'native_measured_dimensions_before_m': list(NATIVE_DIMENSIONS_BEFORE_M),
        'predicted_dimensions_after_m': [dimension * SCALE_RATIO for dimension in NATIVE_DIMENSIONS_BEFORE_M],
        'gripper_pinch_proxy_width_m': 0.10,
        'grasp_and_insertion_verified': False,
        'support_reseating': 'Existing active_layout local_raycast alignment retained unchanged.',
        'variant_scope': 'All non-original variants; selected tabletop has the revised semantics.',
        'original_variant': 'Empty patches retained as the schema-required native baseline.',
    }
    marker = {
        'uniform_scale_ratio': SCALE_RATIO,
        'expected_container_change': {'before': 'container_1', 'after': 'container_0'},
        'expected_answer_change': {'before': 'medium packing box', 'after': 'small packing box'},
        'evidence': evidence,
        'native_validation_status': 'pending',
        'native_validation_run': False,
        'physical_success_verified': False,
    }
    for definition in (revised_task, revised_scene):
        definition['version'] += 1
    revised_task.setdefault('repair', {})[REVISION] = deepcopy(marker)
    changes = {
        'revision': REVISION,
        **deepcopy(marker),
        'versions': {
            'task': {'before': task['version'], 'after': revised_task['version']},
            'scene': {'before': scene['version'], 'after': revised_scene['version']},
        },
        'variant_scales': scale_changes,
    }
    return revised_task, revised_scene, changes


def _check_active(catalogue_path: Path) -> dict:
    """Validate the current selected definitions without changing release files."""
    import yaml

    from wia.tasks.loader import UniqueLoader
    from wia.tasks.schema import CategoryTaskDefinition, SceneDefinition

    root = Path(__file__).resolve().parents[1]
    entries = [row for row in json.loads(catalogue_path.read_text())['tasks'] if row['alias'] == 'T22']
    if len(entries) != 1:
        raise ValueError('The catalogue must contain T22 exactly once.')
    paths = [root / entries[0][key] for key in ('task_file', 'scene_file')]
    task, scene = [yaml.load(path.read_text(), Loader=UniqueLoader) for path in paths]
    snapshots = deepcopy((task, scene))
    revised_task, revised_scene, changes = revise(task, scene)
    CategoryTaskDefinition.model_validate(revised_task)
    SceneDefinition.model_validate(revised_scene)
    assert (task, scene) == snapshots, 'Input definitions changed.'
    assert revised_task['scene'] == revised_scene['scene']
    assert revised_task['private']['bindings'] == revised_scene['bindings']
    assert revised_task['public'] == task['public'], 'Public objective or answer options changed.'
    for old, new in ((task, revised_task), (scene, revised_scene)):
        restored = deepcopy(new)
        restored['version'] = old['version']
        restored['scene'] = deepcopy(old['scene'])
        if old is task:
            restored['repair'].pop(REVISION)
            if 'repair' not in old:
                restored.pop('repair')
            restored['private']['evaluation']['parameters']['expected_containers']['target'] = 'container_1'
            restored['private']['evaluation']['parameters']['expected_answers']['target'] = 'medium packing box'
        assert restored == old, 'A field outside the reviewed revision changed.'
        for name, variant in new['scene']['variants'].items():
            if name == 'original':
                assert variant == old['scene']['variants'][name]
                assert variant['patches'] == []
                continue
            scales = [patch for patch in variant['patches'] if _is_target_scale(patch)]
            assert len(scales) == 1
            before = old['scene']['variants'][name]['patches']
            assert scales[0]['multiplier'] == prod(patch['multiplier'] for patch in before if _is_target_scale(patch)) * SCALE_RATIO
            assert [patch for patch in variant['patches'] if not _is_target_scale(patch)] == [
                patch for patch in before if not _is_target_scale(patch)
            ], 'Container, furniture, poses or support layout changed.'
    try:
        revise(revised_task, revised_scene)
    except ValueError as error:
        assert 'already been applied' in str(error)
    else:
        raise AssertionError('Repeated application was not refused.')
    return {'validation': 'passed', 'mode': 'CPU-only in-memory validation',
        'task_file': str(paths[0]), 'scene_file': str(paths[1]), 'changes': changes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-active', action='store_true', required=True)
    parser.add_argument('--catalogue', type=Path, default=Path(__file__).resolve().parents[1]
        / 'explore_to_achieve/wia_v1/release/catalogue.json')
    args = parser.parse_args(argv)
    print(json.dumps(_check_active(args.catalogue), indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
