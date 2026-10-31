"""Check the CPU-only WIA snapshot publisher without native or server work.

Run: python -B -m pytest -q -p no:cacheprovider tests/test_wia_review_publisher.py
All temporary definitions, captures and output files stay in pytest's temp root.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

import wia.authoring.commands.build_wia_review as publisher


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture_inputs(root, *, updates=None, extras=None, rows=None):
    if rows is None:
        rows = [{
            'wia_id': 'wia_v1_t42', 'document_alias': 'T42',
            'primary_family': 'AP', 'primary_subclass': 'AP-3',
            'implementation_state': 'selected_definition',
            'source_esi_categories': ['Physical Dynamics/Inclined Plane'],
            'task': {'task_file': 'original_task.yaml'},
            'scene': {'selected_scene_file': 'original_scene.yaml'},
            'role': 'diagnostic_candidate', 'tags': ['passive_dynamics'],
            'reassessment': {'reason': 'Native qualification remains pending.'},
            'evidence': {'reference_execution': {'status': 'PASS'}},
        }]
    public = {'objective': 'Compare the three items.', 'success_criteria': 'Deliver the selected spare.',
              'constraints': ['Keep the apparatus unchanged.'], 'environment': 'A ramp and receiver.'}
    for prefix in ('original', 'candidate'):
        task = {'id': prefix, 'public': public, 'source': {'json': '/home/never/read/source.json'},
                'private': {'answer': 'Never a public field.'}}
        (root / f'{prefix}_task.yaml').write_text(yaml.safe_dump(task), encoding='utf-8')
        (root / f'{prefix}_scene.yaml').write_text('id: ' + prefix + '\n', encoding='utf-8')
    write_json(root / 'registry.json', {'rows': rows})
    write_json(root / 'evidence.json', {'schema': publisher.EVIDENCE_SCHEMA,
                                      'row_updates': updates or {}, 'extra_rows': extras or []})
    return public


def build(root):
    return publisher.build_snapshot('registry.json', 'evidence.json', project_root=root,
                                    generated_at='2026-10-23T00:00:00+00:00')


def test_all_61_registry_rows_and_13_unbuilt_proposals_have_reviewable_variants(tmp_path):
    registry = json.loads(publisher.DEFAULT_REGISTRY.read_text(encoding='utf-8'))
    expected_ids = [row['wia_id'] for row in registry['rows']]
    for row in registry['rows']:
        if row['implementation_state'] == 'selected_definition':
            row['task']['task_file'] = 'original_task.yaml'
            row['scene']['selected_scene_file'] = 'original_scene.yaml'
    fixture_inputs(tmp_path, rows=registry['rows'])
    snapshot = build(tmp_path)
    assert [row['id'] for row in snapshot['rows']] == expected_ids
    summary = snapshot['summary']
    assert summary['proposal_count'] == summary['row_count'] == 61
    assert summary['unbuilt_row_count'] == summary['registry_unbuilt_row_count'] == 13
    assert summary['family_counts'] == {'AP': 10, 'NT': 8, 'GF': 9, 'SR': 12, 'GG': 14, 'TM': 8}
    assert summary['authored_variant_count'] == summary['rows_with_images_count'] == 0
    assert summary['reference_success_row_count'] == 0
    for row in snapshot['rows']:
        variant, = row['variants']
        if row['implementation_state'] == 'unbuilt':
            assert variant['id'] == 'proposal'
            assert variant['missing_definition'] is True
            assert variant['objective'] is None
            assert variant['definition_status']['task']['available'] is False
            assert row['status'] == variant['status'] == 'unbuilt'
        else:
            assert variant['id'] == 'original_selected'
            assert variant['missing_definition'] is False
            assert row['status'] == variant['status'] == 'definition_only'


@pytest.mark.parametrize('scene_exists', [True, False])
def test_authored_definitions_update_current_count_without_rewriting_registry(tmp_path, scene_exists):
    rows = [{'wia_id': 'wia_v1_d1_1', 'document_alias': 'D1-1',
             'implementation_state': 'unbuilt', 'primary_family': 'GF'}]
    updates = {'wia_v1_d1_1': {'status': 'definition_only', 'variants': [{
        'id': 'd1_1_seed101_v1', 'task_file': 'candidate_task.yaml',
        'scene_file': 'candidate_scene.yaml', 'status': 'definition_only'}]}}
    fixture_inputs(tmp_path, rows=rows, updates=updates)
    if not scene_exists:
        (tmp_path / 'candidate_scene.yaml').unlink()
    snapshot = build(tmp_path)
    row, = snapshot['rows']
    assert row['registry_implementation_state'] == 'unbuilt'
    assert row['implementation_state'] == ('authored_definition' if scene_exists else 'unbuilt')
    assert snapshot['summary']['unbuilt_row_count'] == (0 if scene_exists else 1)
    assert snapshot['summary']['registry_unbuilt_row_count'] == 1
    assert snapshot['summary']['reference_success_row_count'] == 0
    assert row['variants'][-1]['id'] == 'proposal'
    assert json.loads((tmp_path / 'registry.json').read_text())['rows'][0]['implementation_state'] == 'unbuilt'


def test_first_completed_capture_becomes_default_without_removing_definition_variants(tmp_path):
    (tmp_path / 'capture').mkdir()
    (tmp_path / 'capture/head.png').write_bytes(b'Explicit review image; never decoded here.')
    updates = {'wia_v1_t42': {'variants': [
        {'id': 'authored_definition', 'task_file': 'candidate_task.yaml',
         'scene_file': 'candidate_scene.yaml', 'status': 'definition_only'},
        {'id': 'native_capture', 'task_file': 'candidate_task.yaml',
         'scene_file': 'candidate_scene.yaml', 'status': 'layout_review',
         'captures': [{'directory': 'capture', 'views': ['head']}]}]}}
    fixture_inputs(tmp_path, updates=updates)
    snapshot = build(tmp_path)
    row, = snapshot['rows']
    assert [variant['id'] for variant in row['variants']] == [
        'native_capture', 'authored_definition', 'original_selected']
    assert row['variants'][1]['status'] == 'definition_only'
    assert snapshot['summary']['reference_success_row_count'] == 0


@pytest.mark.parametrize('new_images_ready', [True, False])
def test_explicit_preferred_version_waits_for_images_and_preserves_old_evidence(tmp_path, new_images_ready):
    (tmp_path / 'capture').mkdir()
    (tmp_path / 'capture/head.png').write_bytes(b'Explicit native review image.')
    variants = [{'id': name, 'task_file': 'candidate_task.yaml', 'scene_file': 'candidate_scene.yaml',
                 'captures': [{'directory': 'capture', 'views': ['head']}] if ready else []}
                for name, ready in [('old', True), ('new', new_images_ready)]]
    fixture_inputs(tmp_path, updates={'wia_v1_t42': {'variants': variants, 'preferred_variant_id': 'new'}})
    snapshot = build(tmp_path)
    row, = snapshot['rows']
    assert row['variants'][0]['id'] == ('new' if new_images_ready else 'old')
    assert {variant['id'] for variant in row['variants']} == {'old', 'new', 'original_selected'}
    assert snapshot['summary']['reference_success_row_count'] == 0


def test_curated_variants_prepend_public_fields_and_only_explicit_view_files_are_indexed(tmp_path, monkeypatch):
    captures = tmp_path / 'capture'
    captures.mkdir()
    for view in ('agentview_rgb', 'initial_overview_rgb', 'unlisted_rgb'):
        (captures / (view + '.png')).write_bytes(b'Not decoded or hashed.')
    (captures / 'nested').mkdir()
    (captures / 'nested' / 'wrist_rgb.png').write_bytes(b'Not recursively discovered.')
    updates = {'wia_v1_t42': {'status': 'reference_verified', 'summary': 'Explicit main assessment.',
        'variants': [{'id': 'ramp_candidate_v2', 'label': 'Controlled candidate',
                      'task_file': 'candidate_task.yaml', 'scene_file': 'candidate_scene.yaml',
                      'status': 'reference_verified', 'evidence_summary': 'Human assessment only.',
                      'evidence': [{'label': 'Trial log', 'path': 'evidence/not-opened.json'}],
                      'captures': [{'directory': 'capture', 'label': 'Initial state', 'role': 'layout_review'}]}]}}
    public = fixture_inputs(tmp_path, updates=updates)
    original_read = Path.read_bytes

    def no_image_reads(path):
        assert path.suffix != '.png', 'Publisher must never decode or hash image bytes.'
        return original_read(path)

    monkeypatch.setattr(Path, 'read_bytes', no_image_reads)
    snapshot = build(tmp_path)
    row, = snapshot['rows']
    candidate, original = row['variants']
    assert [candidate['id'], original['id']] == ['ramp_candidate_v2', 'original_selected']
    assert {field: candidate[field] for field in publisher.PUBLIC_FIELDS} == public
    assert 'private' not in candidate and 'answer' not in candidate
    assert row['role'] == 'diagnostic_candidate' and row['tags'] == ['passive_dynamics']
    assert candidate['evidence'][0]['path'] == 'evidence/not-opened.json'
    assert {image['path'] for image in candidate['images']} == {
        str(captures / 'initial_overview_rgb.png'), str(captures / 'agentview_rgb.png')}
    assert all(image['role'] == 'layout_review' and publisher.TOKEN.fullmatch(image['id'])
               for image in candidate['images'])
    assert 'wrist_rgb' in candidate['capture_checks'][0]['missing_views']
    assert snapshot['summary']['rows_with_images_count'] == 1
    assert snapshot['summary']['authored_variant_count'] == 1
    assert snapshot['summary']['reference_success_row_count'] == 1
    assert original['status'] == 'definition_only'


def test_missing_definitions_and_images_never_escalate_status_from_registry_ready_or_pass(tmp_path):
    fixture_inputs(tmp_path)
    registry = json.loads((tmp_path / 'registry.json').read_text())
    registry['rows'][0]['task']['task_file'] = 'missing_task.yaml'
    registry['rows'][0]['status'] = 'ready'
    write_json(tmp_path / 'registry.json', registry)
    snapshot = build(tmp_path)
    row, = snapshot['rows']
    original, = row['variants']
    assert row['status'] == original['status'] == 'definition_only'
    assert original['objective'] is None and original['missing_definition'] is True
    assert original['definition_status']['task']['error'].startswith('FileNotFoundError:')
    assert snapshot['summary']['reference_success_row_count'] == 0


def test_extra_design_row_is_separate_from_61_membership_and_uses_only_authored_public_text(tmp_path):
    extra = {'id': 'wia_v1_ap4_gap', 'document_alias': 'AP-4', 'primary_family': 'AP',
             'primary_subclass': 'AP-4', 'title': 'Proposed control', 'implementation_state': 'unbuilt',
             'status': 'design_only', 'summary': 'Explicit additional design, not an implemented task.',
             'variants': [{'id': 'proposal', 'label': 'Design proposal', 'task_file': '', 'scene_file': '',
                           'objective': 'Caller-authored design objective.', 'status': 'design_only'}]}
    fixture_inputs(tmp_path, extras=[extra])
    snapshot = build(tmp_path)
    row = snapshot['rows'][-1]
    assert row['proposal_membership'] is False
    assert row['variants'][0]['objective'] == 'Caller-authored design objective.'
    assert row['variants'][0]['missing_definition'] is True
    assert snapshot['summary']['proposal_count'] == 1
    assert snapshot['summary']['extra_row_count'] == 1
    assert snapshot['summary']['authored_variant_count'] == 0


@pytest.mark.parametrize('channel', ['task_file', 'scene_file', 'capture', 'evidence', 'symlink'])
def test_home_paths_are_rejected_before_any_definition_or_image_read(tmp_path, monkeypatch, channel):
    variant = {'id': 'candidate', 'task_file': 'candidate_task.yaml', 'scene_file': 'candidate_scene.yaml'}
    forbidden = '/home/never/read/fixture.yaml'
    if channel in ('task_file', 'scene_file'):
        variant[channel] = forbidden
    elif channel == 'capture':
        variant['captures'] = [{'directory': '/home/never/read'}]
    elif channel == 'evidence':
        variant['evidence'] = [{'path': forbidden}]
    else:
        (tmp_path / 'escape.yaml').symlink_to(forbidden)
        variant['task_file'] = 'escape.yaml'
    fixture_inputs(tmp_path, updates={'wia_v1_t42': {'variants': [variant]}})
    original_read = Path.read_bytes

    def guarded(path):
        assert not str(path).startswith('/home/')
        return original_read(path)

    monkeypatch.setattr(Path, 'read_bytes', guarded)
    with pytest.raises(ValueError, match='outside'):
        build(tmp_path)


def test_fingerprint_ignores_status_evidence_and_time_but_tracks_task_or_scene_content(tmp_path):
    fixture_inputs(tmp_path)
    first = build(tmp_path)['rows'][0]['variants'][0]['fingerprint']
    write_json(tmp_path / 'evidence.json', {'schema': publisher.EVIDENCE_SCHEMA,
        'row_updates': {'wia_v1_t42': {'status': 'reference_verified', 'summary': 'Changed assessment.'}}})
    later = publisher.build_snapshot('registry.json', 'evidence.json', project_root=tmp_path,
                                      generated_at='2030-01-01T00:00:00+00:00')
    assert later['rows'][0]['variants'][0]['fingerprint'] == first
    with (tmp_path / 'original_scene.yaml').open('a') as stream:
        stream.write('version: 2\n')
    assert build(tmp_path)['rows'][0]['variants'][0]['fingerprint'] != first


def test_proposal_public_design_changes_invalidate_feedback_but_status_and_evidence_do_not(tmp_path):
    proposal = {'id': 'proposal', 'task_file': '', 'scene_file': '',
                'objective': 'Original design.', 'success_criteria': 'Original criterion.',
                'constraints': ['Original constraint.'], 'environment': 'Original environment.'}
    extra = {'id': 'wia_v1_ap4_gap', 'primary_family': 'AP', 'variants': [proposal]}
    fixture_inputs(tmp_path, extras=[extra])
    first = build(tmp_path)['rows'][-1]['variants'][0]['fingerprint']
    curation = json.loads((tmp_path / 'evidence.json').read_text())
    changed = curation['extra_rows'][0]['variants'][0]
    changed.update(status='design_reviewed', evidence_summary='New human assessment.',
                   evidence=[{'label': 'Assessment', 'path': 'not-opened.json'}])
    write_json(tmp_path / 'evidence.json', curation)
    assert build(tmp_path)['rows'][-1]['variants'][0]['fingerprint'] == first
    for field, value in (('objective', 'Revised design.'), ('success_criteria', 'Revised criterion.'),
                         ('constraints', ['Revised constraint.']), ('environment', 'Revised environment.')):
        version = deepcopy(curation)
        version['extra_rows'][0]['variants'][0][field] = value
        write_json(tmp_path / 'evidence.json', version)
        assert build(tmp_path)['rows'][-1]['variants'][0]['fingerprint'] != first


def test_atomic_publication_preserves_reviews_and_refuses_unrelated_output(tmp_path):
    fixture_inputs(tmp_path)
    snapshot = build(tmp_path)
    reviews = tmp_path / 'reviews.json'
    reviews.write_text('{"human": "keep"}', encoding='utf-8')
    target = publisher.publish_snapshot('review/snapshot.json', snapshot, project_root=tmp_path)
    assert json.loads(target.read_text()) == snapshot
    assert not list(target.parent.glob('*.tmp'))
    replacement = deepcopy(snapshot)
    replacement['generated_at'] = 'later'
    publisher.publish_snapshot('review/snapshot.json', replacement, project_root=tmp_path)
    assert json.loads(target.read_text())['generated_at'] == 'later'
    assert reviews.read_text() == '{"human": "keep"}'
    with pytest.raises(ValueError, match='never write reviews'):
        publisher.publish_snapshot('reviews.json', snapshot, project_root=tmp_path)
    (tmp_path / 'unrelated.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='unrelated'):
        publisher.publish_snapshot('unrelated.json', snapshot, project_root=tmp_path)


@pytest.mark.parametrize('error', ['duplicate_variant', 'unknown_row', 'reserved_variant', 'unknown_view'])
def test_ambiguous_curation_fails_instead_of_dropping_rows(tmp_path, error):
    variant = {'id': 'candidate', 'task_file': '', 'scene_file': ''}
    updates = {'wia_v1_t42': {'variants': [variant]}}
    if error == 'duplicate_variant':
        updates['wia_v1_t42']['variants'].append(deepcopy(variant))
    elif error == 'unknown_row':
        updates = {'unknown': {}}
    elif error == 'reserved_variant':
        variant['id'] = 'original_selected'
    else:
        variant['captures'] = [{'directory': 'capture', 'views': ['../../unexpected']}]
    fixture_inputs(tmp_path, updates=updates)
    with pytest.raises(ValueError):
        build(tmp_path)


def test_supplements_append_variants_and_preserve_main_curation_and_default_behavior(tmp_path):
    main = {'id': 'main_curated', 'task_file': 'candidate_task.yaml', 'scene_file': 'candidate_scene.yaml'}
    fixture_inputs(tmp_path, updates={'wia_v1_t42': {'status': 'physical_partial',
        'summary': 'Main assessment must stay unchanged.', 'variants': [main]}})
    baseline = build(tmp_path)
    assert publisher.build_snapshot('registry.json', 'evidence.json', supplements=(), project_root=tmp_path,
        generated_at='2026-10-23T00:00:00+00:00') == baseline
    for number in (1, 2):
        write_json(tmp_path / f'supplement{number}.json', {'schema': publisher.EVIDENCE_SCHEMA,
            'row_updates': {'wia_v1_t42': {'variants': [{'id': f'native_layout_{number}',
                'task_file': 'original_task.yaml', 'scene_file': 'original_scene.yaml',
                'status': 'native_images_ready', 'capture_review': {'physical_success_inferred': False}}]}}})
    snapshot = publisher.build_snapshot('registry.json', 'evidence.json',
        supplements=['supplement1.json', 'supplement2.json'], project_root=tmp_path)
    row, = snapshot['rows']
    assert row['status'] == 'physical_partial'
    assert row['summary'] == 'Main assessment must stay unchanged.'
    assert [variant['id'] for variant in row['variants']] == [
        'main_curated', 'native_layout_1', 'native_layout_2', 'original_selected']
    assert row['variants'][0] == baseline['rows'][0]['variants'][0]
    assert row['variants'][1]['capture_review']['physical_success_inferred'] is False
    assert snapshot['summary']['reference_success_row_count'] == 0
    assert snapshot['provenance']['supplements'] == ['supplement1.json', 'supplement2.json']


@pytest.mark.parametrize('conflict', ['variant', 'repeat_file', 'row_status', 'row_summary',
                                     'row_public', 'extra_row', 'unknown_row', 'top_title', 'duplicate_json'])
def test_supplements_reject_conflicts_instead_of_overriding_main_fields(tmp_path, conflict):
    fixture_inputs(tmp_path, updates={'wia_v1_t42': {'variants': [{'id': 'main_curated'}]}})
    value = {'schema': publisher.EVIDENCE_SCHEMA,
             'row_updates': {'wia_v1_t42': {'variants': [{'id': 'native_capture'}]}}}
    supplements = ['supplement.json']
    if conflict == 'variant':
        value['row_updates']['wia_v1_t42']['variants'][0]['id'] = 'main_curated'
    elif conflict == 'repeat_file':
        supplements.append('supplement.json')
    elif conflict.startswith('row_'):
        field = conflict.removeprefix('row_')
        value['row_updates']['wia_v1_t42'][field] = 'Must not override.'
    elif conflict == 'extra_row':
        value['extra_rows'] = [{'id': 'extra'}]
    elif conflict == 'unknown_row':
        value['row_updates'] = {'unknown': {'variants': [{'id': 'native_capture'}]}}
    elif conflict == 'top_title':
        value['title'] = 'Must not replace main title.'
    write_json(tmp_path / 'supplement.json', value)
    if conflict == 'duplicate_json':
        (tmp_path / 'supplement.json').write_text(
            '{"schema":"wia_review_evidence_v1","row_updates":{"wia_v1_t42":{"variants":[]},'
            '"wia_v1_t42":{"variants":[]}}}', encoding='utf-8')
    with pytest.raises(ValueError):
        publisher.build_snapshot('registry.json', 'evidence.json', supplements=supplements, project_root=tmp_path)


def test_explicit_native_view_names_are_allowed_without_expanding_default_capture_selection(tmp_path):
    fixture_inputs(tmp_path)
    directory = tmp_path / 'native'
    directory.mkdir()
    for view in publisher.ALLOWED_VIEWS:
        (directory / (view + '.png')).write_bytes(b'No image decoding.')
    images, _ = publisher._images([{'directory': 'native', 'views': sorted(publisher.ALLOWED_VIEWS)}],
                                  'native_capture', tmp_path)
    assert len(images) == len(publisher.ALLOWED_VIEWS)
    assert any(image['path'].endswith('/object_detail_31.png') for image in images)
    assert any(image['path'].endswith('/phase_2_global_diagonal.png') for image in images)
    assert 'object_detail_32' not in publisher.ALLOWED_VIEWS
    defaults, _ = publisher._images([{'directory': 'native'}], 'native_capture', tmp_path)
    assert len(defaults) == len(publisher.DEFAULT_VIEWS) == 9


def test_temporary_exclusions_hide_rows_and_recount_without_destroying_evidence(tmp_path):
    excluded_aliases = ['D1-1', 'D1-2', 'D1-3', 'D2-1', 'D2-2']
    aliases = [*excluded_aliases, *(f'T{number:02d}' for number in range(56))]
    rows = [{'wia_id': f'row_{index}', 'document_alias': alias,
             'primary_family': 'GF' if index < 5 else 'AP',
             'implementation_state': 'unbuilt' if index < 5 else 'selected_definition',
             'task': {'task_file': 'original_task.yaml'},
             'scene': {'selected_scene_file': 'original_scene.yaml'}}
            for index, alias in enumerate(aliases)]
    extras = [{'id': f'candidate_{number}', 'document_alias': f'C{number:02d}',
               'primary_family': 'AP', 'release_admitted': False,
               'variants': [{'id': 'candidate_review', 'task_file': 'candidate_task.yaml',
                             'scene_file': 'candidate_scene.yaml'}]} for number in range(19)]
    updates = {f'row_{index}': {'status': 'reference_verified', 'variants': [{
        'id': 'native_review', 'task_file': 'candidate_task.yaml', 'scene_file': 'candidate_scene.yaml',
        'captures': [{'directory': 'capture', 'views': ['head']}]}]} for index in range(5)}
    fixture_inputs(tmp_path, rows=rows, extras=extras, updates=updates)
    (tmp_path / 'capture').mkdir()
    (tmp_path / 'capture/head.png').write_bytes(b'Historical screenshot bytes.')
    (tmp_path / 'reviews.json').write_text('{"feedback":"Preserve every historical decision."}')
    before = build(tmp_path)
    assert before['summary']['proposal_count'] == 61
    assert before['summary']['row_count'] == 80
    curation = json.loads((tmp_path / 'evidence.json').read_text())
    curation['temporarily_excluded_aliases'] = excluded_aliases
    write_json(tmp_path / 'evidence.json', curation)
    protected = {path: path.read_bytes() for path in tmp_path.rglob('*') if path.is_file()}
    after = build(tmp_path)
    summary = after['summary']
    assert summary['proposal_count'] == 56
    assert summary['row_count'] == 75
    assert summary['extra_row_count'] == 19
    assert summary['family_counts'] == {'AP': 56}
    assert summary['extra_family_counts'] == {'AP': 19}
    assert summary['active_family_counts'] == {'AP': 75}
    assert summary['unbuilt_row_count'] == 19
    assert summary['registry_unbuilt_row_count'] == 0
    assert summary['reference_success_row_count'] == summary['rows_with_images_count'] == 0
    assert summary['authored_variant_count'] == 19
    assert after['provenance']['temporarily_excluded_aliases'] == excluded_aliases
    assert not set(excluded_aliases) & {row['document_alias'] for row in after['rows']}
    assert all(row['proposal_membership'] is False and row['release_admitted'] is False
               for row in after['rows'][-19:])
    publisher.publish_snapshot('review/snapshot.json', after, project_root=tmp_path)
    assert all(path.read_bytes() == content for path, content in protected.items())
    assert len(json.loads((tmp_path / 'registry.json').read_text())['rows']) == 61
    assert json.loads((tmp_path / 'evidence.json').read_text())['row_updates'] == updates


def test_temporary_exclusions_also_support_explicit_extra_aliases(tmp_path):
    extras = [{'id': 'extra', 'document_alias': 'EXTRA', 'primary_family': 'AP',
               'variants': [{'id': 'candidate_review'}]}]
    fixture_inputs(tmp_path, extras=extras)
    curation = json.loads((tmp_path / 'evidence.json').read_text())
    curation['temporarily_excluded_aliases'] = ['EXTRA']
    write_json(tmp_path / 'evidence.json', curation)
    snapshot = build(tmp_path)
    assert snapshot['summary']['proposal_count'] == snapshot['summary']['row_count'] == 1
    assert snapshot['summary']['extra_row_count'] == 0
    assert snapshot['summary']['extra_family_counts'] == {}
    assert snapshot['provenance']['temporarily_excluded_aliases'] == ['EXTRA']


@pytest.mark.parametrize('exclusions', [None, 'T42', [42], ['T42', 'T42'], ['UNKNOWN']])
def test_temporary_exclusions_reject_wrong_types_duplicates_and_unknown_aliases(tmp_path, exclusions):
    fixture_inputs(tmp_path)
    curation = json.loads((tmp_path / 'evidence.json').read_text())
    curation['temporarily_excluded_aliases'] = exclusions
    write_json(tmp_path / 'evidence.json', curation)
    with pytest.raises(ValueError, match='(excluded_aliases|unknown document aliases)'):
        build(tmp_path)


def test_excluded_rows_still_validate_curation_and_supplement_conflicts(tmp_path):
    fixture_inputs(tmp_path, updates={'wia_v1_t42': {'variants': [{'id': 'main_curated'}]}})
    curation = json.loads((tmp_path / 'evidence.json').read_text())
    curation['temporarily_excluded_aliases'] = ['T42']
    write_json(tmp_path / 'evidence.json', curation)
    write_json(tmp_path / 'supplement.json', {'schema': publisher.EVIDENCE_SCHEMA,
        'row_updates': {'wia_v1_t42': {'variants': [{'id': 'main_curated'}]}}})
    with pytest.raises(ValueError, match='Conflicting duplicate'):
        publisher.build_snapshot('registry.json', 'evidence.json', supplements=['supplement.json'],
                                 project_root=tmp_path)
