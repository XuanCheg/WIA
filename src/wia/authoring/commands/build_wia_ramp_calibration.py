"""Build four pending single-factor T42 ramp-material calibration candidates.

Run: python -B scripts/build_wia_ramp_calibration.py [--check] [--output-dir PATH]
This CPU generator never initializes simulation or asserts a winning candidate.


Canonical invocation: python -m wia.authoring.commands.build_wia_ramp_calibration --help
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import yaml

from wia.resources import ROOT

from wia.tasks.loader import UniqueLoader, task_definition
from wia.tasks.schema import SceneDefinition
from wia.tasks.sources import resolve_source

BASELINE_RELATIVE = Path('Physical Dynamics/Inclined Plane/house_single_floor/redesign_20261020/category_88ce9d62d891ae0ba628a6e9.yaml')
BASELINE_TASK = ROOT / 'explore_to_achieve/tasks' / BASELINE_RELATIVE
BASELINE_SCENE = ROOT / 'explore_to_achieve/scenes' / BASELINE_RELATIVE
DEFAULT_OUTPUT = ROOT / 'explore_to_achieve/wia_v1/ramp_calibration'
MULTIPLIERS = (0., .1, .35, 1.)
ORIGINAL_FRICTION = {'static_friction': 2.744, 'dynamic_friction': 1.7137}
PUBLIC_TASK_ID = 'wia_ramp_material_calibration_v1'
SCHEMA = 'wia_ramp_material_calibration_candidates_v1'
MANIFEST_NAME = 'manifest.private.json'


def read_definition(path):
    return yaml.load(Path(path).read_text(encoding='utf-8'), Loader=UniqueLoader)


def ramp_hooks(definition):
    """Locate exactly the existing before/after active-layout material hooks."""
    hooks = [patch for variant in definition['scene']['variants'].values()
             for patch in variant['patches'] if patch.get('operation') == 'category_initialization'
             and patch.get('module') == 'core.esi_converters.active_layout']
    if len(hooks) != 2 or {hook['phase'] for hook in hooks} != {'before_construction', 'after_postprocessing'}:
        raise ValueError('The T42 baseline must retain exactly its two active-layout initialization hooks.')
    return hooks


def baseline_definitions():
    task, scene = read_definition(BASELINE_TASK), read_definition(BASELINE_SCENE)
    task_definition(task)
    SceneDefinition.model_validate(scene)
    evaluation = task['private']['evaluation']
    if (evaluation['module'], evaluation['operation']) != (
            'core.esi_converters.inclined_comparison', 'controlled_trials_then_spare_delivery'):
        raise ValueError('The T42 baseline evaluator has changed; review the calibration contract.')
    parameters = evaluation['parameters']
    if {key: parameters[key] for key in ORIGINAL_FRICTION} != ORIGINAL_FRICTION:
        raise ValueError('The T42 baseline material differs from the declared original coefficients.')
    if parameters['trial_duration_s'] != 3. or parameters['arrival_tie_tolerance_s'] != .15:
        raise ValueError('The T42 trial duration or tie tolerance has changed.')
    if parameters['source_ramp_geometry_and_friction_preserved'] is not True:
        raise ValueError('The T42 baseline no longer declares preserved source ramp geometry and friction.')
    if task['scene'] != scene['scene'] or task['source'] != scene['source']:
        raise ValueError('The selected T42 task and scene initialization contracts disagree.')
    for definition in (task, scene):
        for hook in ramp_hooks(definition):
            if hook['parameters']['layout']['friction']['ramp'] != ORIGINAL_FRICTION:
                raise ValueError('An active-layout ramp material differs from the baseline evaluator.')
    return task, scene


def build_candidates():
    """Return private metadata and schema-validated definitions with one material factor."""
    baseline_task, baseline_scene = baseline_definitions()
    source_identity = baseline_task['source']['json']
    source_path = resolve_source(source_identity)
    pairs, entries = [], []
    for index, multiplier in enumerate(MULTIPLIERS):
        condition = f'material_{index:03d}'
        friction = {key: value * multiplier for key, value in ORIGINAL_FRICTION.items()}
        task, scene = deepcopy(baseline_task), deepcopy(baseline_scene)
        task.update(id=PUBLIC_TASK_ID, version=1, status='pending')
        scene.update(id='wia_ramp_private_' + condition, version=1, status='pending')
        for definition in (task, scene):
            for hook in ramp_hooks(definition):
                hook['parameters']['layout']['friction']['ramp'].update(friction)
        parameters = task['private']['evaluation']['parameters']
        parameters.update(friction)
        parameters['source_ramp_geometry_and_friction_preserved'] = multiplier == 1.
        task_definition(task)
        SceneDefinition.model_validate(scene)
        task_file, scene_file = f'{condition}/task.yaml', f'{condition}/scene.yaml'
        pairs.append({'task_file': task_file, 'scene_file': scene_file, 'task': task, 'scene': scene})
        entries.append({
            'private_condition_id': condition, 'task_file': task_file, 'scene_file': scene_file,
            'ramp_material_multiplier': multiplier, 'ramp_material_coefficients': friction,
            'source_identity': source_identity,
            'private_provenance': {
                'source_ramp_geometry_preserved': True,
                'source_ramp_geometry_and_friction_preserved': multiplier == 1.,
                'original_task_file': BASELINE_TASK.relative_to(ROOT).as_posix(),
                'original_scene_file': BASELINE_SCENE.relative_to(ROOT).as_posix(),
            },
            'evidence': {'native_calibration': 'pending', 'trial_release_repeatability': 'pending',
                         'legal_robot_success': 'pending', 'winner_reversal': 'not_established'},
        })
    manifest = {
        'schema': SCHEMA, 'version': 1, 'status': 'pending_native_material_calibration',
        'privacy': 'private_configuration_and_provenance_not_policy_input',
        'artifact_path_base': 'manifest_directory', 'original_path_base': 'repository_root',
        'public_task_identity': {'id': PUBLIC_TASK_ID, 'version': 1},
        'selection_method': 'exact_task_and_scene_paths_required_shared_public_task_id',
        'factor': {'name': 'ramp_material_multiplier', 'independent_dimensions': 1,
                   'definition': 'One scalar multiplies both original ramp friction coefficients; their nonzero ratio is fixed.',
                   'multipliers': list(MULTIPLIERS), 'original_coefficients': ORIGINAL_FRICTION,
                   'zero_condition': 'Both coefficients are zero; their ratio is undefined at zero.'},
        'hypothesis': 'At lower ramp friction, bowl sliding may overtake the rolling ball. This is untested; no winner or reversal is asserted.',
        'native_experiments_run_by_generator': False, 'agent_experiments_run_by_generator': False,
        'publication_ready': False,
        'source_inputs': {name: {'path': path.relative_to(ROOT).as_posix(),
                                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                          for name, path in (('task', BASELINE_TASK), ('scene', BASELINE_SCENE), ('source', source_path))},
        'preserved_protocol': {
            'source_identity': source_identity, 'public_objective': 'unchanged',
            'geometry_poses_assets_spares': 'unchanged', 'seed_cameras_controller': 'unchanged',
            'trial_duration_s': baseline_task['private']['evaluation']['parameters']['trial_duration_s'],
            'arrival_tie_tolerance_s': baseline_task['private']['evaluation']['parameters']['arrival_tie_tolerance_s'],
            'orientation_release_recovery': 'unchanged',
        },
        'entries': entries,
    }
    return manifest, pairs


def render_artifacts():
    manifest, pairs = build_candidates()
    artifacts = {MANIFEST_NAME: json.dumps(manifest, sort_keys=True, indent=2) + '\n'}
    for pair in pairs:
        for kind in ('task', 'scene'):
            artifacts[pair[kind + '_file']] = yaml.safe_dump(pair[kind], sort_keys=False, allow_unicode=False)
    artifacts['README.md'] = """# AP-3 pending ramp-material calibration candidates

These four path-selected task/scene pairs reuse T42 and vary one scalar material multiplier applied jointly to the original ramp static and dynamic friction coefficients. All geometry, poses, assets, spares, seeds, cameras, controllers, public wording, release/orientation/recovery rules, trial duration and tie tolerance remain fixed. Their nonzero coefficient ratio is fixed; the zero condition sets both coefficients to zero.

The task ID and version exposed by the full loader public payload are identical across all conditions. Scene IDs and condition paths are private. Select each task and scene by its exact path; the public task ID does not select a unique condition. The source identity remains unchanged. Definitions remain pending and require diagnostic-mode loading until qualified.

`manifest.private.json` is private experiment configuration. It records condition multipliers, actual coefficients, original paths and source hashes. Do not send it, condition paths or scene IDs to the policy. The numerical material fields in task private evaluation and scene initialization necessarily configure the simulator; they are not public task fields.

The hypothesis is that bowl sliding may overtake the rolling ball at lower ramp friction. No winner, winner reversal, admissible trial or legal robot success is asserted. Native material calibration, trial-release repeatability and complete legal robot execution remain pending. Changing the tie tolerance or selecting a winning condition is outside this generator.

Regenerate with the [builder](/local_home2/zhouxinyang/models/Show-Harness-Explore/scripts/build_wia_ramp_calibration.py): `python -B scripts/build_wia_ramp_calibration.py`. Append `--check` for a deterministic no-write comparison or use `--output-dir /tmp/wia-ramp-candidates` for a separate copy. Existing unrelated files are never deleted or overwritten. The generator performs no native, GPU or Agent experiments and leaves the original task, scene, source and WIA registry unchanged.

Run the [CPU bookkeeping tests](/local_home2/zhouxinyang/models/Show-Harness-Explore/tests/test_wia_ramp_calibration.py) with `python -B -m pytest -q -p no:cacheprovider tests/test_wia_ramp_calibration.py`. These checks validate the single-factor configuration and public-payload equality; they do not qualify physical outcomes.
"""
    return artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check', action='store_true', help='Compare generated bytes without writing.')
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    artifacts = render_artifacts()
    if args.check:
        differences = [name for name, text in artifacts.items()
                       if not (output / name).is_file() or (output / name).read_text(encoding='utf-8') != text]
        if differences:
            print('Ramp calibration candidates differ or are missing: ' + ', '.join(differences), file=sys.stderr)
            return 1
        print(f'Ramp calibration candidates are current: {output}')
        return 0
    existing = [name for name in artifacts if (output / name).exists()]
    if existing:
        manifest_file = output / MANIFEST_NAME
        try:
            previous = json.loads(manifest_file.read_text(encoding='utf-8'))
        except (OSError, ValueError, UnicodeError) as exc:
            raise ValueError('Refusing to overwrite files without an owned ramp-calibration manifest.') from exc
        if not isinstance(previous, dict) or previous.get('schema') != SCHEMA:
            raise ValueError('Refusing to overwrite an unrelated ramp-calibration output.')
    for name, text in artifacts.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    print(f'Wrote four pending ramp-material calibration pairs: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
