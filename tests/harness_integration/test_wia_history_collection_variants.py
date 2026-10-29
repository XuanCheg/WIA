"""Check the two authored pantry variants without native simulator execution.

Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -B -m
pytest -q -p no:cacheprovider tests/test_wia_history_collection_variants.py
"""
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import yaml

from wia.authoring.converters import temporal_understanding_agent_observation as converter
from wia.authoring.converters.wia_history_collection import HistoryCollectionEvaluator
from core.observations import PublicTaskEvent
from wia.tasks.loader import load_scene, load_task
from wia.tasks.schema import resolve_episode_defaults
from wia.tasks.wia_history_collection import public_collection_query, seeded_collection_query
import wia.authoring.commands.build_wia_history_collection_variants as builder


@pytest.fixture
def package(tmp_path):
    output = tmp_path / 'history_collection'
    definitions, manifest, rendered = builder.build_package(output)
    return output, definitions, manifest, rendered


def parent_pair(entry):
    return tuple(yaml.safe_load(builder.repo_path(entry['parent_' + kind + '_file']).read_text())
                 for kind in ('task', 'scene'))


def test_exact_nine_asset_inventory_supports_cyclic_slots_and_original_root_heights(package):
    _, definitions, manifest, rendered = package
    assert len(definitions) == len(manifest['entries']) == 2 and len(rendered) == 5
    assert manifest['schema'] == 'wia_variant_manifest_v1'
    for definition, entry in zip(definitions, builder.PARENTS):
        task = definition['task']
        parent_task, _ = parent_pair(entry)
        original_layout = builder._layout_hooks(parent_task)[0]['parameters']['layout']
        original = {obj['name']: obj for obj in original_layout['objects']}
        hooks = builder._layout_hooks(task)
        assert hooks[0]['parameters'] == hooks[1]['parameters']
        layout = hooks[0]['parameters']['layout']
        objects = {obj['name']: obj for obj in layout['objects']}
        p = task['private']['evaluation']['parameters']
        items = [objects[key] for key in p['target_keys']]
        assert len(p['target_keys']) == len(set(p['target_keys'])) == 9
        assert Counter((obj['category'], obj['model']) for obj in items) == {
            ('hamburger', 'fyswch'): 3, ('hamburger_bun', 'fqxumh'): 3, ('club_sandwich', 'ablfyd'): 3}
        supports = p['apparatus']['target_support_bindings']
        assert set(supports) == set(p['target_keys'])
        assert {row['name']: row['support'] for row in layout['supported_items']} == supports
        assert Counter(supports.values()) == {f'pantry_shelf_{station}_deck': 3 for station in range(1, 4)}
        for station in range(1, 4):
            deck_name = f'pantry_shelf_{station}_deck'
            deck = objects[deck_name]
            # Independently recover local shelf XY from the authored world poses.
            qx, qy, qz, qw = deck['orientation']
            yaw = math.atan2(2 * (qw*qz + qx*qy), 1 - 2 * (qy*qy + qz*qz))
            station_items = [obj for obj in items if supports[obj['name']] == deck_name]
            station_items.sort(key=lambda obj: math.cos(yaw) * (obj['position'][0] - deck['position'][0])
                               + math.sin(yaw) * (obj['position'][1] - deck['position'][1]))
            expected_categories = [builder.ASSETS[(station - 1 + slot) % 3][0] for slot in range(3)]
            assert [obj['category'] for obj in station_items] == expected_categories
            for slot, obj in enumerate(station_items):
                dx, dy = [obj['position'][axis] - deck['position'][axis] for axis in (0, 1)]
                assert math.cos(yaw)*dx + math.sin(yaw)*dy == pytest.approx([-.18, 0., .18][slot])
                assert -math.sin(yaw)*dx + math.cos(yaw)*dy == pytest.approx(-.10)
                donor_index = next(index for index, asset in enumerate(builder.ASSETS, 1) if asset[0] == obj['category'])
                source = original[f'pantry_item_{donor_index}']
                source_deck = original[f'pantry_shelf_{donor_index}_deck']
                source_height = source['position'][2] - source_deck['position'][2] - source_deck['scale'][2]/2
                assert obj['position'][2] - deck['position'][2] - deck['scale'][2]/2 == pytest.approx(source_height)
                assert obj['orientation'] == deck['orientation'] and obj['scale'] == [1., 1., 1.]
                assert obj['fixed_base'] is obj['visual_only'] is False
                assert task['private']['bindings'][obj['name']] == {
                    field: obj[field] for field in ('name', 'category', 'model')}
            assert supports[f'pantry_item_{station}'] == deck_name


def test_all_noninventory_world_runtime_budgets_and_scientific_thresholds_preserved(package):
    _, definitions, manifest, _ = package
    changed_parameters = {'target_keys', 'public_target_labels', 'target_categories', 'query_rule',
        'require_history_subset', 'query_min_category_instances', 'query_min_total', 'query_max_total',
        'history_collection_revision', 'followup_rule', 'apparatus'}
    for definition, parent, record in zip(definitions, builder.PARENTS, manifest['entries']):
        task, scene = definition['task'], definition['scene']
        parent_task, parent_scene = parent_pair(parent)
        assert task['source'] == scene['source'] == parent_task['source'] == parent_scene['source']
        assert task['defaults'] == parent_task['defaults']
        assert scene['seed'] == parent_scene['seed'] == 0
        assert scene['runtime'] == parent_scene['runtime']
        assert task['requirements'] == parent_task['requirements']
        assert task['qualification'] == parent_task['qualification']
        assert scene['qualification'] == parent_scene['qualification']
        before_p = parent_task['private']['evaluation']['parameters']
        after_p = task['private']['evaluation']['parameters']
        assert {k: v for k, v in after_p.items() if k not in changed_parameters} == {
            k: v for k, v in before_p.items() if k not in changed_parameters}
        assert after_p['release_stability_steps'] == 15 and after_p['followup_seed'] == 7
        before_patches = deepcopy(parent_task['scene']['variants']['adapted']['patches'])
        after_patches = deepcopy(task['scene']['variants']['adapted']['patches'])
        assert len(before_patches) == len(after_patches)
        for before, after in zip(before_patches, after_patches):
            if before['module'] == builder.INITIALIZER:
                original_layout = before['parameters']['layout']
                updated_layout = after['parameters']['layout']
                original_objects = {obj['name']: obj for obj in original_layout.pop('objects')}
                updated_objects = {obj['name']: obj for obj in updated_layout.pop('objects')}
                for name, obj in original_objects.items():
                    if name not in {'pantry_item_1', 'pantry_item_2', 'pantry_item_3'}:
                        assert updated_objects[name] == obj
                original_layout.pop('supported_items')
                updated_layout.pop('supported_items')
            assert before == after
        assert task['scene']['variants']['original'] == parent_task['scene']['variants']['original']
        assert record['episode_defaults'] == resolve_episode_defaults(parent_task).model_dump(mode='json')
        assert record['episode_defaults']['max_steps'] == 60
        assert record['episode_defaults']['wall_timeout_s'] == 5400.
        assert 'unqualified' in record['budget_status']


def test_current_historical_query_contract_public_prompt_and_input_dispatch(package, tmp_path):
    _, definitions, _, _ = package
    for definition in definitions:
        task = definition['task']
        evaluation = deepcopy(task['private']['evaluation'])
        p = evaluation['parameters']
        assert evaluation['operation'] == 'history_multi_retrieval_v1'
        assert evaluation['module'] == builder.MODULE
        assert p['query_rule'] == 'earliest_per_category' and p['require_history_subset'] is True
        assert (p['query_min_category_instances'], p['query_min_total'], p['query_max_total']) == (2, 1, 3)
        assert Counter(p['target_categories'].values()) == {
            'hamburgers with filling': 3, 'plain buns': 3, 'club sandwiches': 3}
        assert set(p['public_target_labels']) == set(p['target_keys'])
        assert len(set(p['public_target_labels'].values())) == 9
        assert set(p['target_keys']) <= set(evaluation['binding_keys']) == set(task['private']['bindings'])
        public = json.dumps(task['public'])
        assert 'all nine' in public and 'three hamburgers' in public and 'three plain buns' in public
        assert 'within each category' in public and 'ANSWER: SURVEY_COMPLETE' in public
        assert 'same requested category' in public and 'Different categories' in public
        assert 'answer' not in public.replace('ANSWER: SURVEY_COMPLETE', '').lower()
        assert not any(label in public for label in p['public_target_labels'].values())
        assert not any(key in public for key in task['private']['bindings'])
        rows = [{'key': key, 'snapshot_id': index + 1, 'action_step': index + 1}
                for index, key in enumerate(p['target_keys'])]
        query = seeded_collection_query(rows, p)
        assert all(request['count'] < 3 for request in query['requests'])
        event = {'event_id': 'fixture_followup', 'kind': 'history_followup',
            'text': public_collection_query(query, p['destination_label'])['text'], 'snapshot_id': 9, 'action_step': 9}
        assert PublicTaskEvent(**event).to_dict() == event
        evaluation['bindings'] = {key: binding['name'] for key, binding in task['private']['bindings'].items()}
        evaluator = converter.create_evaluator(None, None, evaluation, tmp_path / task['id'])
        assert isinstance(evaluator, HistoryCollectionEvaluator)


def test_manifest_exact_provenance_pending_flags_and_real_cpu_loaders(package):
    output, definitions, manifest, rendered = package
    assert builder.build_package(output)[2] == rendered
    for record, definition in zip(manifest['entries'], definitions):
        assert record['variant_id'].endswith('_historical_multi_collection_v2')
        for field in ('parent_task', 'parent_scene', 'source', 'builder'):
            assert record[field + '_sha256'] == builder.sha256(Path(record[field + '_file']).read_bytes())
        assert record['task_sha256'] == builder.sha256(rendered[definition['task_file']])
        assert record['scene_sha256'] == builder.sha256(rendered[definition['scene_file']])
        assert record['source_identity'] == record['task_source_identity'] == record['scene_source_identity']
        assert record['task_source_sha256'] == record['scene_source_sha256'] == record['source_sha256']
        assert Path(record['parent_task_file']).resolve() not in rendered
        assert record['initialization_status'] == 'authored_not_native_validated'
        for flag in builder.NATIVE_FLAGS:
            assert manifest[flag] is record[flag] is definition['task']['repair'][flag] is False
    assert builder.main(['--output-dir', str(output)]) == 0
    for definition in definitions:
        task = load_task(definition['task_file'], variant='adapted', run_mode='diagnostic')
        scene = load_scene(definition['scene_file'], variant='adapted', run_mode='diagnostic')
        assert task.definition.scene == scene.definition.scene
        assert task.definition.private.bindings == scene.definition.bindings
        assert task.seed == scene.seed == 0


def test_cli_check_idempotency_parent_immutability_and_no_differing_overwrite(package):
    output, definitions, _, rendered = package
    inputs = {builder.repo_path(entry['parent_' + kind + '_file']) for entry in builder.PARENTS for kind in ('task', 'scene')}
    inputs.update(load_task(path, variant='adapted', run_mode='diagnostic').source_path
                  for path in list(inputs) if '/tasks/' in str(path))
    before = {path: path.read_bytes() for path in inputs}
    assert builder.main(['--output-dir', str(output), '--check']) == 1
    assert not output.exists()
    assert builder.main(['--output-dir', str(output)]) == 0
    mtimes = {path: path.stat().st_mtime_ns for path in rendered}
    assert builder.main(['--output-dir', str(output), '--check']) == 0
    assert builder.main(['--output-dir', str(output)]) == 0
    assert {path: path.stat().st_mtime_ns for path in rendered} == mtimes
    changed = definitions[0]['task_file']
    changed.write_bytes(changed.read_bytes() + b'\n')
    assert builder.main(['--output-dir', str(output), '--check']) == 1
    with pytest.raises(ValueError, match='Refusing to overwrite'):
        builder.main(['--output-dir', str(output)])
    assert {path: path.read_bytes() for path in inputs} == before
