"""Build the Stage 0 WIA proposal overlay without initializing a simulator.

Run: python -B scripts/build_wia_registry.py --check
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml

from wia.tasks.sources import ROOT, resolve_source


RESEARCH = Path('docs/research/world_interactive_agency_v1')
INPUTS = {
    'recheck': RESEARCH / 'task_recheck_61.csv',
    'inventory': RESEARCH / 'inventory_reconciliation.json',
    'taxonomy': RESEARCH / 'source_documents/02_Taxonomy/World_Interactive_Agency_Taxonomy_v1.yaml',
    'selection': Path('explore_to_achieve/lists/corev2.json'),
    'source_suggestions': RESEARCH / 'source_documents/06_Independent_Review/World_Interactive_Agency_V1_61_Task_Review.csv',
}
DEFAULT_OUTPUT = ROOT / 'explore_to_achieve/wia_v1/registry.json'
EXPECTED_FAMILIES = {'AP': 10, 'NT': 8, 'GF': 9, 'SR': 12, 'GG': 14, 'TM': 8}
EVIDENCE_DIMENSIONS = ('human_layout_review', 'native_initialization', 'physical_scorer',
                       'reference_execution', 'scientific_controls')
SUBCLASS_TAGS = {
    'AP-1': ['prescribed_order', 'physical_execution'],
    'AP-2': ['support_geometry', 'released_stability'],
    'AP-3': ['controlled_physical_trials', 'outcome_conditioned_selection'],
    'NT-1': ['topology', 'observed_region_grounding'],
    'NT-2': ['supplied_route', 'ordered_arrival'],
    'NT-3': ['observed_anchor_grounding', 'region_relation'],
    'GF-1': ['containment_affordance', 'partial_insertion'],
    'GF-2': ['dimension_constraints', 'legal_insertion_path'],
    'GF-3': ['intrinsic_dimension', 'physical_assignment'],
    'SR-1': ['geometric_arrangement', 'released_relation'],
    'SR-2': ['native_contact', 'separation_regulation'],
    'SR-3': ['reflection', 'fixed_reference', 'physical_reconfiguration'],
    'GG-1': ['target_identity', 'physical_collection'],
    'GG-2': ['deformable_occlusion', 'physical_retrieval'],
    'GG-3': ['reflection', 'target_correspondence', 'physical_delivery'],
    'GG-4': ['appearance_condition', 'containment_state'],
    'TM-1': ['observation_history', 'delayed_target_action'],
    'TM-2': ['world_state_history', 'contingent_action'],
}


def _read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def _unique(rows, key, label):
    result = {}
    for row in rows:
        value = row.get(key)
        if not value or value in result:
            raise ValueError(f'{label} contains a missing or duplicate {key}: {value!r}')
        result[value] = row
    return result


def _relative(path, root):
    """Resolve known resource migrations and enforce repository containment."""
    from wia.resources import resolve_migrated_path
    try:
        path = resolve_migrated_path(path, root=root).relative_to(Path(root).resolve())
    except ValueError as exc:
        raise ValueError(f'Definition path is outside the repository or changed: {path}') from exc
    if '..' in path.parts:
        raise ValueError(f'Definition path escapes the repository: {path}')
    return path.as_posix()


def _load_yaml(path):
    # Reuse the task loader's duplicate-key rejection without loading task runtimes.
    from wia.tasks.loader import UniqueLoader
    value = yaml.load(path.read_text(encoding='utf-8'), Loader=UniqueLoader)
    if not isinstance(value, dict):
        raise ValueError(f'Expected a mapping: {path}')
    return value


def _taxonomy(taxonomy, recheck):
    families = _unique(taxonomy['families'], 'id', 'Taxonomy families')
    subclasses, memberships, gaps = {}, {}, []
    for family_id, family in families.items():
        for subclass in family['subcategories']:
            key = subclass['id']
            if key in subclasses or not key.startswith(family_id + '-'):
                raise ValueError(f'Duplicate or mismatched taxonomy subclass: {key}')
            subclasses[key] = {'id': key, 'name': subclass['name'], 'family': family_id,
                               'design_gap': subclass['design_gap']}
            aliases = subclass['current_candidate_task_ids']
            if subclass['design_gap']:
                if aliases:
                    raise ValueError(f'Design gap unexpectedly contains proposal rows: {key}')
                gaps.append(key)
            for alias in aliases:
                if alias in memberships:
                    raise ValueError(f'Duplicate taxonomy membership: {alias}')
                memberships[alias] = key
    if set(memberships) != set(recheck):
        raise ValueError('Taxonomy and recheck aliases have a missing or extra mapping.')
    for alias, row in recheck.items():
        subclass = row['proposed_subclass']
        if memberships[alias] != subclass or subclasses[subclass]['family'] != row['proposed_family']:
            raise ValueError(f'Taxonomy mapping disagrees with recheck: {alias}')
    if gaps != ['AP-4']:
        raise ValueError('Expected AP-4 to remain a design gap outside the 61-row proposal.')
    return list(subclasses.values()), gaps


def _source_relative(path, root):
    """Keep registry pointer spelling while resolving actual upstream files canonically."""
    from wia.resources import ESI_ROOT
    path, root = Path(path), Path(root)
    legacy = root / 'ESI-Bench'
    if legacy.is_dir() and legacy.resolve() == ESI_ROOT:
        try:
            return (Path('ESI-Bench') / path.relative_to(ESI_ROOT)).as_posix()
        except ValueError:
            pass
    return _relative(path, root)


def _source(identity, root):
    """Resolve only explicit identities through the existing bounded resolver."""
    path = resolve_source(identity)
    raw = json.loads(path.read_text(encoding='utf-8'))
    parts = Path(identity).parts
    result = {
        'identity': identity, 'path': _source_relative(path, root),
        'esi_category': parts[0], 'esi_subcategory': parts[1],
        'source_scene': raw.get('scene'), 'source_room': raw.get('room'),
    }
    metadata = raw.get('metadata_json', {})
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    parent = metadata.get('authored_revision', {}).get('parent_source')
    if parent:
        parent_path = resolve_source(parent)
        result['explicit_parent_source'] = {'identity': parent, 'path': _source_relative(parent_path, root)}
    return result


def _observation_contract(implemented):
    from wia.tasks.observation_spec import MOUNTED_RGB_VIEWS
    return {
        'id': 'eight_mounted_plus_frozen_initial_overview_v1',
        'applicability': 'current_public_contract' if implemented else 'intended_contract_unbuilt',
        'mounted_rgb_views': list(MOUNTED_RGB_VIEWS),
        'initial_overview': {'view': 'initial_overview_rgb', 'refresh_policy': 'frozen_at_episode_start'},
        'actual_target_visibility': 'not_assessed',
        'evidence_scope': 'sensor_interface_only_not_a_visibility_or_information_necessity_claim',
        'implementation_files': ['core/robot_cameras.py', 'core/observations.py', 'core/agent/image_inputs.py'],
    }


def _scorer(definition):
    evaluation = definition.get('private', {}).get('evaluation')
    if not evaluation:
        return {'status': 'not_configured', 'type': None, 'module': None, 'operation': None,
                'version': None, 'explicit_version_fields': {},
                'reason': 'No private evaluation is declared; loader ready and reuse labels do not configure a physical scorer.'}
    parameters = evaluation.get('parameters', {})
    versions = {key: parameters[key] for key in ('scoring_version', 'revision', 'task_revision', 'schema', 'version')
                if key in parameters}
    return {'status': 'configured_not_validated', 'type': evaluation.get('type'),
            'module': evaluation.get('module'), 'operation': evaluation.get('operation'),
            'version': parameters.get('scoring_version'), 'explicit_version_fields': versions,
            'reason': 'Configuration is not evidence of physical scorer correctness or complete execution.'}


def _evidence(selected, scorer):
    """Separate recorded initialization/capture claims from complete action evidence."""
    capture = selected.get('current_capture', {})
    pointers = []
    for receipt in capture.get('receipts', []):
        pointers.append({'pointer': receipt['file'], 'sha256_recorded': receipt.get('sha256'),
                         'scope': 'current_registry_capture_reference_not_revalidated',
                         'job_id': capture.get('job_id')})
    for key in ('current_native_qualification_receipt', 'current_capture_receipt', 'evidence_file'):
        if selected.get(key):
            pointers.append({'pointer': selected[key], 'scope': key + '_not_revalidated'})
    for key in ('independent_reset_validation', 'passive_physics_validation'):
        if selected.get(key):
            item = selected[key]
            pointers.append({'pointer': item['file'], 'sha256_recorded': item.get('sha256'),
                             'scope': key + '_not_robot_execution', 'recorded_claims': item})
    # Pointer strings, including historical /home paths, remain verbatim and are never opened.
    base = INPUTS['selection'].as_posix() + '#candidate_id=' + selected['candidate_id']
    historical = {'pointer': base, 'scope': 'selection_record_only'}
    has_layout = bool(selected.get('camera_audit') or selected.get('visual_inspection_notes'))
    return {
        'human_layout_review': {
            'status': 'inconclusive' if has_layout else 'not_assessed',
            'pointers': [historical],
            'recorded_camera_audit': selected.get('camera_audit'),
            'recorded_visual_inspection_notes': selected.get('visual_inspection_notes'),
            'reason': 'Historical visual notes do not establish current layout approval or complete action.'},
        'native_initialization': {
            'status': 'inconclusive' if pointers else 'not_assessed', 'pointers': pointers,
            'recorded_initialization_accepted': capture.get('initialization_accepted'),
            'recorded_capture_state': capture.get('state'),
            'recorded_input_identity': capture.get('initial_agent_bundle', {}).get('capture_input_identity'),
            'reason': 'Registry receipts are preserved without revalidating current definition compatibility.'},
        'physical_scorer': {
            'status': 'inconclusive' if scorer['status'] != 'not_configured' else 'not_assessed',
            'configuration_status': scorer['status'], 'pointers': [historical],
            'reason': 'No independent physical scorer validation is established by this overlay.'},
        'reference_execution': {
            'status': 'not_assessed', 'pointers': [historical],
            'recorded_execution_qualified': capture.get('execution_qualified', selected.get('current_execution_qualified')),
            'recorded_final_release_qualified': selected.get('final_release_qualified'),
            'reason': 'Captures, passive probes, ready and setup acceptance are not complete legal reference trajectories.'},
        'scientific_controls': {
            'status': 'not_assessed', 'pointers': [],
            'reason': 'Tags and proposed control roles do not validate causal or capability claims.'},
    }


def _implemented(selected, root):
    task_file = _relative(selected['task_file'], root)
    scene_file = _relative(selected['scene_file'], root)
    definition = _load_yaml(root / task_file)
    scene = _load_yaml(root / scene_file)
    source = _source(definition['source']['json'], root)
    scene_source = _source(scene['source']['json'], root)
    # Rigid tasks can bind a criterion-specific QA source to a separate approved physical source.
    if scene['source']['json'] != source['identity'] and definition.get('kind') != 'rigid_containment_task':
        raise ValueError(f'Selected task/scene source mismatch: {selected["candidate_id"]}')
    original_sources = [_source(identity, root) for identity in selected['source_identities']]
    if not original_sources:
        raise ValueError(f'Selected task lacks source identity: {selected["candidate_id"]}')
    scene_name, workspace = scene_source['source_scene'], scene_source['source_room']
    if not isinstance(scene_name, str) or not isinstance(workspace, str):
        scene_name, workspace = None, None
    scorer = _scorer(definition)
    capabilities = definition.get('requirements', {}).get('capabilities')
    init_modules = sorted({patch['module'] for variant in scene.get('scene', {}).get('variants', {}).values()
                           for patch in variant.get('patches', [])
                           if patch.get('operation') == 'category_initialization' and patch.get('module')})
    return {
        'task': {'id': definition['id'], 'version': definition.get('version', 1),
                 'task_file': task_file, 'kind': definition.get('kind'), 'loader_status': definition.get('status'),
                 'physical_goal': definition.get('goal'),
                 'allow_remove_after_verification': definition.get('allow_remove_after_verification')},
        'scene': {'id': scene['id'], 'version': scene.get('version', 1), 'selected_scene_file': scene_file,
                  'task_declared_scene_file': definition.get('scene_file'),
                  'scene_name': scene_name, 'workspace': workspace,
                  'identity_basis': 'current_selected_definition_source_metadata_not_native_measurement',
                  'actual_workspace_geometry': 'unknown',
                  'source_identity': scene['source']['json']},
        'source': source, 'selected_scene_source': scene_source, 'original_sources': original_sources,
        'source_esi_categories': sorted({item['esi_category'] + '/' + item['esi_subcategory']
                                        for item in [source, scene_source, *original_sources]}),
        'evaluator': scorer,
        'public_observation_contract': _observation_contract(True),
        'allowed_capabilities': capabilities,
        'capability_basis': 'explicit_task_requirements' if capabilities is not None else 'not_declared_by_task_definition',
        'engineering_reuse': {
            'scene': {'status': 'existing_selected_definition', 'path': scene_file},
            'converter': {'status': ('existing_configuration_not_validated' if init_modules or scorer['module']
                                     else 'not_explicitly_configured'), 'initialization_modules': init_modules,
                          'evaluation_module': scorer['module']},
            'scorer': {'status': scorer['status'], 'module': scorer['module'], 'operation': scorer['operation']},
            'evidence': {'status': 'registry_pointers_only_not_revalidated'},
        },
        'evidence': _evidence(selected, scorer),
    }


def _suggestion(review, root):
    url = review['source_url']
    path = unquote(urlparse(url).path)
    marker = '/dataset/json_clean/'
    suggested = _source(path.split(marker, 1)[1], root) if marker in path else None
    return {'status': 'source_suggestion_not_an_initialized_scene', 'source': suggested,
            'scene_suggestion': review['scene_id'], 'room_suggestion': review['room'],
            'source_url': url if marker in path else None,
            'provenance': INPUTS['source_suggestions'].as_posix() + '#task_id=' + review['task_id']}


def _unbuilt(review, root):
    return {
        'task': None, 'scene': {'id': None, 'version': None, 'selected_scene_file': None,
                              'scene_name': None, 'workspace': None, 'identity_basis': 'unbuilt',
                              'actual_workspace_geometry': 'unknown'},
        'source': None, 'original_sources': [], 'source_esi_categories': [],
        'source_suggestion': _suggestion(review, root),
        'public_observation_contract': _observation_contract(False),
        'evaluator': {'status': 'not_configured', 'type': None, 'module': None, 'operation': None,
                      'version': None, 'explicit_version_fields': {}, 'reason': 'Unbuilt proposal.'},
        'allowed_capabilities': None, 'capability_basis': 'unbuilt_not_declared',
        'engineering_reuse': {key: {'status': 'not_implemented'} for key in ('scene', 'converter', 'scorer', 'evidence')},
        'evidence': {key: {'status': 'not_assessed', 'pointers': [], 'reason': 'Unbuilt proposal.'}
                     for key in EVIDENCE_DIMENSIONS},
    }


def build_registry(*, root=ROOT, recheck_path=None, inventory_path=None, taxonomy_path=None,
                   selection_path=None, suggestions_path=None):
    """Return a deterministic overlay; raise for incomplete or duplicate mappings."""
    root = Path(root).resolve()
    overrides = dict(zip(INPUTS, (recheck_path, inventory_path, taxonomy_path, selection_path, suggestions_path)))
    paths = {key: Path(overrides[key]) if overrides[key] is not None else root / value
             for key, value in INPUTS.items()}
    recheck = _unique(_read_csv(paths['recheck']), 'document_task_id', 'Recheck')
    _unique(list(recheck.values()), 'canonical_candidate_or_design_id', 'Recheck')
    inventory = json.loads(paths['inventory'].read_text(encoding='utf-8'))
    taxonomy = _load_yaml(paths['taxonomy'])
    subclasses, gaps = _taxonomy(taxonomy, recheck)
    selection = json.loads(paths['selection'].read_text(encoding='utf-8'))
    selected = _unique(selection['tasks'], 'candidate_id', 'CoreV2 selection')
    _unique(selection['tasks'], 'task_file', 'CoreV2 selection')
    suggestions = _unique(_read_csv(paths['source_suggestions']), 'task_id', 'Source suggestions')
    if set(suggestions) != set(recheck):
        raise ValueError('Source suggestions and recheck have a missing or extra alias mapping.')
    rows, retained = [], set()
    for alias, row in recheck.items():
        if row['currently_selected'] not in ('True', 'False'):
            raise ValueError(f'Invalid selected state: {alias}')
        implemented = row['currently_selected'] == 'True'
        candidate = row['canonical_candidate_or_design_id']
        if implemented:
            if candidate not in selected:
                raise ValueError(f'Missing selected candidate mapping: {alias}')
            for key in ('task_file', 'scene_file'):
                if _relative(row[key], root) != _relative(selected[candidate][key], root):
                    raise ValueError(f'Selected {key} disagrees with recheck: {alias}')
            payload = _implemented(selected[candidate], root)
            if payload['task']['kind'] != row['task_kind']:
                raise ValueError(f'Task kind disagrees with recheck: {alias}')
            declared_scorer = payload['evaluator']
            expected_module = ('not_configured' if declared_scorer['status'] == 'not_configured'
                               else declared_scorer['module'])
            expected_operation = (payload['task']['physical_goal'] if declared_scorer['status'] == 'not_configured'
                                  else declared_scorer['operation'])
            for key, expected in (('evaluator_module', expected_module), ('evaluator_operation', expected_operation)):
                if (row[key] or None) != expected:
                    raise ValueError(f'Evaluator mapping disagrees with recheck: {alias}')
            retained.add(candidate)
        else:
            if candidate in selected or row['task_file'] or row['scene_file'] or row['task_kind'] != 'unbuilt_design':
                raise ValueError(f'Unbuilt row has an implementation mapping: {alias}')
            payload = _unbuilt(suggestions[alias], root)
        subclass = row['proposed_subclass']
        role = ('execution_control' if subclass in ('AP-1', 'NT-2') else
                'correct_stop_control' if alias in ('T30', 'T32', 'T41') else
                'control_source_candidate' if alias == 'R-A2' else 'diagnostic_candidate')
        rows.append({
            'wia_id': 'wia_v1_' + alias.lower().replace('-', '_'), 'document_alias': alias,
            'canonical_candidate_id': candidate, 'implementation_state': 'selected_definition' if implemented else 'unbuilt',
            'primary_family': row['proposed_family'], 'primary_subclass': subclass,
            'classification_status': 'provisional_not_construct_validated',
            'tags': SUBCLASS_TAGS[subclass], 'role': role, 'role_status': 'proposed_not_validated',
            'historical_reuse_label': row['original_reuse_level'],
            'reassessment': {'decision': row['reassessment'], 'reason': row['reason']},
            **payload,
        })
    controls = []
    for candidate, entry in selected.items():
        if candidate in retained:
            continue
        if (entry['category'], entry['subcategory']) != ('Metric Comparison', 'Dimensional Size'):
            raise ValueError(f'Unexpected nonproposal selected control: {candidate}')
        controls.append({'canonical_candidate_id': candidate, 'role': 'legacy_dimensional_size_control',
                         'proposal_membership': False, 'selection_unchanged': True, **_implemented(entry, root)})
    declared_controls = {row['candidate_id']: _relative(row['task_file'], root)
                         for row in inventory['selected_not_in_proposal']}
    actual_controls = {row['canonical_candidate_id']: row['task']['task_file'] for row in controls}
    if actual_controls != declared_controls:
        raise ValueError('Legacy controls disagree with the reconciled selected inventory.')
    families = dict(Counter(row['primary_family'] for row in rows))
    counts = {'proposal_count': len(rows), 'retained_selected_count': len(retained),
              'unbuilt_proposal_count': sum(row['implementation_state'] == 'unbuilt' for row in rows),
              'selected_count': len(selected), 'legacy_control_count': len(controls),
              'proposed_family_counts': families,
              'proposed_subclass_counts': dict(Counter(row['primary_subclass'] for row in rows)),
              'configured_scorer_count': sum(row['evaluator']['status'] == 'configured_not_validated' for row in rows),
              'selected_without_configured_scorer_count': sum(row['implementation_state'] == 'selected_definition'
                  and row['evaluator']['status'] == 'not_configured' for row in rows)}
    for key in ('proposal_count', 'retained_selected_count', 'unbuilt_proposal_count', 'selected_count', 'proposed_family_counts'):
        if counts[key] != inventory[key]:
            raise ValueError(f'Derived {key} disagrees with the inventory.')
    if families != EXPECTED_FAMILIES or len(rows) != 61 or len(selected) != 51 or len(controls) != 3:
        raise ValueError('Stage 0 counts differ from the 61-proposal/51-selected baseline.')
    migrations = inventory['family_migrations']
    expected_migrations = {('T15', 'GF', 'GG'), ('T16', 'GF', 'GG'), ('T17', 'GF', 'GG'),
                           ('D2-1', 'SR', 'GF'), ('D2-2', 'SR', 'GF')}
    if {(m['task'], m['old'], m['proposed']) for m in migrations} != expected_migrations or len(migrations) != 5:
        raise ValueError('Expected exactly the five declared taxonomy migrations.')
    for migration in migrations:
        if recheck[migration['task']]['proposed_family'] != migration['proposed']:
            raise ValueError('A declared taxonomy migration disagrees with its proposal row.')
    return {
        'schema': 'world_interactive_agency_registry_v1', 'version': '1.0.0-stage0',
        'status': 'proposal_overlay_not_release', 'path_base': 'repository_root',
        'baseline': {'code_commit': inventory['code_baseline'], 'review_date': inventory['review_date'],
                     'corev2_schema': selection['schema'], 'corev2_status': selection['status'],
                     'corev2_execution_qualified_count_recorded': selection['execution_qualified_count']},
        'source_inputs': {key: {'path': value.as_posix(), 'sha256': hashlib.sha256(paths[key].read_bytes()).hexdigest()}
                          for key, value in INPUTS.items()},
        'metadata': counts,
        'evidence_policy': {'native_experiments_run_by_builder': False, 'agent_experiments_run_by_builder': False,
                            'historical_receipts_opened_by_builder': False, 'publication_ready': False,
                            'independent_dimensions': list(EVIDENCE_DIMENSIONS),
                            'missing_evidence': 'not_assessed_or_inconclusive_never_inferred_pass'},
        'taxonomy': {'schema': taxonomy['schema'], 'version': taxonomy['version'],
                     'classification_status': 'provisional_not_construct_validated',
                     'primary_membership_disjoint': True, 'subclasses': subclasses,
                     'design_gaps_outside_proposal': gaps},
        'family_migrations': migrations, 'rows': rows, 'legacy_controls': controls,
    }


def serialize_registry(registry):
    """Use stable UTF-8 JSON bytes with a trailing newline and no timestamps."""
    return json.dumps(registry, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
