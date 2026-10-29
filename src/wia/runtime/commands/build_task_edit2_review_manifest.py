"""Build provisional TaskEdit2 coverage from explicit current manifests and exact evidence.

Run: PYTHONDONTWRITEBYTECODE=1 python -B src/wia/runtime/commands/build_task_edit2_review_manifest.py
No simulator, queue, Review UI, source definition or final Core selection is changed.
"""
from __future__ import annotations

from wia.resources import ROOT as WIA_RUNTIME_ROOT

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = WIA_RUNTIME_ROOT
sys.path.insert(0, str(ROOT))

import yaml
from wia.tasks.loader import UniqueLoader, task_definition
from wia.tasks import loader as task_loader
from wia.tasks.schema import SceneDefinition

BASELINE = Path('/home/zhouxinyang/esi_category_conversion/core_revision_20261018/core_candidates_for_review.json')
CURRENT_INPUTS = (
    'missing_definitions/missing_definitions_manifest.json',
    'capacity_inventory/batch_manifest.json',
    'reflection_v2/batch_manifest.json',
    'inclined_apparatus_candidates_v2/batch_manifest.json',
    'perception/manifest.json',
    'grounding/grounding_manifest.json',
    'temporal/manifest.json',
    'mapping_worlds/mapping_worlds_manifest.json',
)
OPTIONAL_INPUTS = ('deformable_candidates_v2/batch_manifest.json', 'temporal_core/manifest.json',
                   'temporal_core_v2/manifest.json', 'temporal_core_v3/manifest.json',
                   'geometry_core/batch_manifest.json', 'initialization_repairs/manifest.json',
                   'capacity_calibrated/batch_manifest.json', 'perception_v3/search_manifest.json',
                   'reflection_v3/batch_manifest.json', 'reflection_v3_height_pilot/batch_manifest.json',
                   'reflection_v4_height_pilot/batch_manifest.json')
ABORT_ORIGIN = 'root_recorded_stop_of_actual_native_runtime_after_fatal_error'
INITIALIZATION_CHECKPOINT_STAGES = ('constructed', 'source_postprocessed', 'adapted')
MOUNTED = ('agentview_rgb', 'wrist_rgb', 'right_wrist_rgb', 'chassis_front_left_rgb',
           'chassis_front_right_rgb', 'chassis_left_rgb', 'chassis_right_rgb', 'chassis_rear_rgb')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def absolute(value, root=ROOT):
    if not value:
        return None
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else Path(root) / path).resolve())


def _dicts(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _dicts(item)


def _members(record):
    """Use only explicit source mapping fields, never category/family similarity."""
    rows = []
    for key in ('source_to_parent', 'source_mappings', 'sources'):
        for value in record.get(key, []) or []:
            if isinstance(value, dict) and value.get('source_identity'):
                rows.append({**deepcopy(value), 'mapping_field': key})
            elif isinstance(value, str) and value.endswith('.json') and len(Path(value).parts) == 5:
                rows.append({'source_identity': value, 'mapping_field': key})
    if record.get('source_identity'):
        rows.append({'source_identity': record['source_identity'], 'source_sha256': record.get('source_sha256'),
                     'mapping_field': 'source_identity'})
    by_source = {}
    for row in rows:
        identity = row['source_identity']
        entry = by_source.setdefault(identity, {'source_identity': identity, 'declared_source_hashes': [], 'mappings': []})
        if row.get('source_sha256') and row['source_sha256'] not in entry['declared_source_hashes']:
            entry['declared_source_hashes'].append(row['source_sha256'])
        if row not in entry['mappings']:
            entry['mappings'].append(row)
    return list(by_source.values())


class Builder:
    def __init__(self, root, source_root):
        self.root, self.source_root = Path(root).resolve(), Path(source_root).resolve()
        self.files, self.ambiguities = {}, []

    def definition(self, filename, kind):
        path = absolute(filename, self.root)
        if not path:
            return {'path': None, 'sha256': None, 'status': 'unavailable', 'reason': 'definition_not_supplied'}, None
        if path in self.files:
            return deepcopy(self.files[path][0]), self.files[path][1]
        if not Path(path).is_file():
            result = {'path': path, 'sha256': None, 'status': 'unavailable', 'reason': 'file_missing'}
            return result, None
        data = Path(path).read_bytes()
        result = {'path': path, 'sha256': hashlib.sha256(data).hexdigest(), 'status': 'schema_invalid'}
        raw = None
        try:
            raw = yaml.load(data, Loader=UniqueLoader)
            if not isinstance(raw, dict):
                raise ValueError('Definition must be a mapping.')
            if kind == 'task':
                task_definition(raw)
            else:
                SceneDefinition.model_validate(raw)
            result.update(status='schema_valid', id=raw['id'], source=deepcopy(raw['source']))
        except (ValueError, TypeError, KeyError, yaml.YAMLError) as error:
            result['reason'] = f'{type(error).__name__}: {error}'[:1500]
            if not isinstance(raw,dict):
                raw=None
        self.files[path] = (deepcopy(result), raw)
        return result, raw

    def source(self, member):
        identity = member['source_identity']
        path = self.source_root / identity
        result = {**deepcopy(member), 'source_file': str(path.resolve()), 'source_sha256': None,
                  'provenance_status': 'unavailable'}
        # A source identity is an exact canonical relative source path.
        if Path(identity).is_absolute() or '..' in Path(identity).parts or len(Path(identity).parts) != 5:
            result['reason'] = 'noncanonical_source_identity'
        elif not path.is_file():
            result['reason'] = 'source_file_missing'
        else:
            result['source_sha256'] = sha256(path)
            declared = member.get('declared_source_hashes', [])
            result['provenance_status'] = ('hash_mismatch' if any(value != result['source_sha256'] for value in declared)
                                           else 'verified_current_source_bytes')
        return result

    def protected_parent_linkage(self, record, task, task_raw, scene, scene_raw, members, *, original):
        """Verify only a protected original Rigid task's explicit parent-scene link.

        This is static loader provenance, not native qualification. The ordinary
        exact_native_match policy deliberately does not accept this classification.
        """
        if (not original or record.get('protected_baseline') is not True
                or not isinstance(task_raw, dict) or task_raw.get('kind') != 'rigid_containment_task'):
            return None
        result = {'status': 'unverified_declared_protected_parent_linkage',
                  'native_qualification_transfer': 'excluded_from_generic_exact_native_match'}
        try:
            if task['status'] != 'schema_valid' or scene['status'] != 'schema_valid':
                raise ValueError('Both protected definitions must validate their actual schemas.')
            root = Path(task_loader.SCENE_ROOT).resolve()
            declared = (root / task_raw['scene_file']).resolve()
            if not declared.is_relative_to(root) or declared != Path(scene['path']).resolve():
                raise ValueError('Declared task.scene_file differs from the exact supplied scene path.')
            question_identity = task_raw['source']['json']
            parent_identity = scene_raw['source']['json']
            if (question_identity == parent_identity or record.get('source') != question_identity
                    or {m['source_identity'] for m in members} != {question_identity}
                    or any(m['provenance_status'] != 'verified_current_source_bytes' for m in members)):
                raise ValueError('Protected linkage requires one exact unchanged original question source.')
            loaded = task_loader.load_task(task['path'], data_root=root)
            loaded.verify_unchanged()
            question_path = (root / question_identity).resolve()
            parent_source = (root / parent_identity).resolve()
            if (Path(loaded.path).resolve() != Path(task['path']).resolve()
                    or Path(loaded.scene.path).resolve() != declared
                    or Path(loaded.question_path).resolve() != question_path
                    or Path(loaded.scene.source_path).resolve() != parent_source
                    or loaded.file_sha256 != task['sha256']
                    or loaded.scene.file_sha256 != scene['sha256']
                    or loaded.question_sha256 != sha256(question_path)
                    or loaded.scene.source_sha256 != sha256(parent_source)
                    or any(m['source_sha256'] != loaded.question_sha256 for m in members)):
                raise ValueError('Loader identities or question/parent source bytes differ from reviewed definitions.')
            question = json.loads(question_path.read_text())
            parent = json.loads(parent_source.read_text())
            targets = tuple(scene_raw.get('target_bindings') or (scene_raw.get('target'),))
            if ((question.get('scene'), question.get('room')) != (loaded.scene_name, loaded.room_name)
                    or (parent.get('scene'), parent.get('room')) != (loaded.scene_name, loaded.room_name)
                    or question.get('question') != task_raw['public']['objective']
                    or tuple(task_raw['targets']) != targets):
                raise ValueError('Original question, same source scene/room, or all target ordering differs.')
            result.update(status='verified_declared_protected_parent_linkage',
                loader='wia.tasks.loader.load_task -> load_rigid_task',
                declared_scene_file=task_raw['scene_file'], resolved_scene_file=str(declared),
                task_file=task['path'], task_sha256=loaded.file_sha256,
                scene_sha256=loaded.scene.file_sha256,
                question_source={'source_identity': question_identity, 'path': str(question_path),
                                 'sha256': loaded.question_sha256},
                parent_scene_source={'source_identity': parent_identity, 'path': str(parent_source),
                                     'sha256': loaded.scene.source_sha256},
                source_scene=loaded.scene_name, source_room=loaded.room_name,
                original_question_verbatim=True, target_bindings_in_order=list(targets),
                protocol_sha256=loaded.protocol_sha256)
        except (ValueError, TypeError, KeyError, AttributeError, OSError, RuntimeError) as error:
            result['reason'] = f'{type(error).__name__}: {error}'[:1500]
        return result

    def variant(self, record, manifest_path, *, original=False):
        task, task_raw = self.definition(record.get('task_file'), 'task')
        scene, scene_raw = self.definition(record.get('scene_file'), 'scene')
        members = _members(record)
        for member in members:
            identity = member['source_identity']
            for document in (record, task_raw, scene_raw):
                for row in _dicts(document):
                    if row.get('source_identity') == identity and row.get('source_sha256'):
                        value = row['source_sha256']
                        if value not in member['declared_source_hashes']:
                            member['declared_source_hashes'].append(value)
        members = [self.source(member) for member in members]
        declared_sources = {m['source_identity'] for m in members}
        definition_sources={}
        source_check = 'verified'
        source_ambiguities = []
        for kind, definition, raw in (('task', task, task_raw), ('scene', scene, scene_raw)):
            raw_source=raw.get('source',{}) if isinstance(raw,dict) else {}
            source_identity=raw_source.get('json') if isinstance(raw_source,dict) else None
            definition_sources[kind]=(self.source({'source_identity':source_identity,'declared_source_hashes':[],
                                                  'mappings':[]}) if source_identity else None)
            if raw is None:
                source_check = 'unavailable'
            elif source_identity not in declared_sources:
                source_check = 'definition_source_mismatch'
                source_ambiguities.append({'candidate_id':record.get('candidate_id',record.get('group_key')),
                    'manifest':manifest_path, 'kind':kind, 'reason':source_check,
                    'definition_source':raw.get('source'), 'explicit_members':sorted(declared_sources)})
        if any(m['provenance_status'] == 'hash_mismatch' for m in members):
            source_check = 'source_hash_mismatch'
        elif source_check == 'verified' and any(m['provenance_status'] == 'unavailable' for m in members):
            source_check = 'unavailable'
        if not members:
            source_check = 'unresolved_manifest_mapping'
            self.ambiguities.append({'manifest':manifest_path, 'candidate_id':record.get('candidate_id',record.get('group_key')),
                                     'reason':'no_explicit_source_mapping'})
        linkage = None
        if source_check == 'definition_source_mismatch':
            linkage = self.protected_parent_linkage(record, task, task_raw, scene, scene_raw, members,
                                                     original=original)
            if linkage and linkage['status'] == 'verified_declared_protected_parent_linkage':
                source_check = linkage['status']
                source_ambiguities = []
        self.ambiguities.extend(source_ambiguities)
        candidate = record.get('candidate_id') or record.get('group_key')
        key = hashlib.sha256(json.dumps([candidate,task['path'],task['sha256'],scene['path'],scene['sha256']],sort_keys=True).encode()).hexdigest()[:24]
        pending = record.get('pending_gates', record.get('native_gates', []))
        pending = ([key for key,value in pending.items() if value is not True and value not in ('verified','accepted')]
                   if isinstance(pending, dict) else list(pending or []))
        declared_checks = {}
        for keyname in ('task_sha256','scene_sha256'):
            expected = record.get(keyname)
            if expected:
                actual = task['sha256'] if keyname == 'task_sha256' else scene['sha256']
                declared_checks[keyname] = {'declared':expected,'actual':actual,'matches':expected==actual}
        return {'variant_key':key,'candidate_id':candidate,'role':'original_baseline' if original else 'explicit_derivative',
            'manifest_records':[{'manifest_file':manifest_path,'record_status':record.get('status',record.get('conversion_status','unknown'))}],
            'task':task,'scene':scene,'sources':members,'source_binding_status':source_check,
            'definition_source_provenance':definition_sources,
            'declared_protected_parent_linkage':linkage,
            'declared_definition_hash_checks':declared_checks,
            'definition_review_status':'unknown',
            'schema_status': 'schema_valid' if task['status']==scene['status']=='schema_valid' else 'unavailable'
                if 'unavailable' in (task['status'],scene['status']) else 'schema_invalid',
            'recorded_variant_status':record.get('status',record.get('conversion_status','unknown')),
            'pending_gates':pending,'recorded_warnings':deepcopy(record.get('warning_codes',[])),
            'native_initialization_status':'unknown','layout_status':'unknown','execution_status':'unavailable',
            'scoring_status':'unavailable','autonomous_agent_performance':False,'native_evidence_ids':[],
            'selection_status':'provisional_user_review','selected_final_variant':False,
            'artifacts':self.explicit_images(record),
            'original_record':deepcopy(record) if original else None}

    def explicit_images(self, record):
        paths = set()
        for row in _dicts(record):
            for value in row.values():
                if isinstance(value,str) and value.lower().endswith('.png') and Path(value).is_absolute() and Path(value).is_file():
                    paths.add(str(Path(value).resolve()))
        return {'explicit_original_or_review_images':sorted(paths),'current_initial_capture_status':'pending'}


def _source_hashes(result):
    values = []
    for row in (result, result.get('initial_condition',{}), result.get('loaded_task',{}),
                result.get('provenance',{}), result.get('loaded_task_provenance',{})):
        if isinstance(row,dict):
            values.extend(row[key] for key in ('source_sha256','esi_source_sha256') if row.get(key))
    return set(values)


def exact_native_match(variant, result, root=ROOT, *, checkpoint_proof=None):
    """Require exact paths and both current file hashes plus exact source provenance."""
    checks = {}
    for kind in ('task','scene'):
        paths={result.get(kind+'_file')}
        hashes={result.get(kind+'_sha256')}
        for name in ('loaded_task','loaded_scene','provenance','loaded_task_provenance'):
            loaded=result.get(name,{})
            if not isinstance(loaded,dict):continue
            paths.add(loaded.get(kind+'_file'))
            hashes.add(loaded.get(kind+'_file_sha256'))
            if name==('loaded_task' if kind=='task' else 'loaded_scene'):
                paths.add(loaded.get('path'))
                hashes.add(loaded.get('file_sha256'))
        paths={absolute(path,root) for path in paths if path}
        hashes={value for value in hashes if value}
        checks[kind+'_path']=bool(variant[kind]['path'] and paths=={variant[kind]['path']})
        checks[kind+'_sha256'] = bool(variant[kind]['sha256'] and hashes=={variant[kind]['sha256']})
    primary = (variant['task'].get('source') or variant['scene'].get('source') or {}).get('json')
    actual_source = next((row['source_sha256'] for row in variant['sources'] if row['source_identity']==primary),None)
    source_hashes = _source_hashes(result)
    if (checkpoint_proof and checkpoint_proof.get('accepted') is True
            and result.get('status') == 'stopped_at_requested_initialization_checkpoint'
            and result.get('diagnostic_checkpoint_complete') is True):
        captured = checkpoint_proof['captured_variant_binding']
        checks['checkpoint_source_identity'] = checkpoint_proof['captured_source_identity'] == primary
        checks['source_sha256'] = bool(actual_source and captured['source_sha256'] == actual_source
            and (not source_hashes or source_hashes == {actual_source}))
    else:
        checks['source_sha256'] = bool(actual_source and source_hashes=={actual_source})
    checks['definition_source_binding'] = variant['source_binding_status']=='verified'
    return all(checks.values()), checks


def initialization_checkpoint_proof(path, result):
    """Validate only the exact requested same-run read-only stage receipt."""
    stage = result.get('checkpoint_stage')
    checks = {'allowed_checkpoint_stage':stage in INITIALIZATION_CHECKPOINT_STAGES}
    receipt_path = path.parent / ('initialization_' + stage + '.json') if checks['allowed_checkpoint_stage'] else None
    checks['same_directory_receipt'] = bool(receipt_path and receipt_path.resolve().parent == path.parent.resolve()
        and receipt_path.is_file())
    captured = {}
    binding = {}
    error = None
    receipt_sha = sha256(receipt_path) if checks['same_directory_receipt'] else None
    try:
        if not checks['same_directory_receipt']:
            raise ValueError('Exact allowed same-directory initialization receipt is unavailable.')
        receipt = json.loads(receipt_path.read_text())
        cameras = receipt['camera_mounts']
        checks['captured_checkpoint_stage'] = (receipt.get('stage') == stage
            and bool(receipt.get('source_constructor'))
            and cameras.get('schema') == 'initial_camera_mount_diagnostic_v1'
            and cameras.get('stage') == stage)
        captured = deepcopy(cameras['provenance'])
        def measured(row, key):
            value = row.get(key,{})
            return value.get('value') if isinstance(value,dict) and value.get('available') is True else None
        captured_sources, source_files = [], []
        for kind in ('task','scene'):
            row = captured.get(kind,{})
            for key, field in (('file',kind+'_file'),('file_sha256',kind+'_sha256')):
                value = measured(row,key)
                binding[field] = value
                checks['captured_'+kind+'_'+key] = bool(value and result.get(field) and
                    (absolute(value) == absolute(result[field]) if key == 'file' else value == result[field]))
            captured_sources.append(measured(row,'source_sha256'))
            filename = measured(row,'source_file')
            source_file = Path(absolute(filename)) if filename else None
            source_files.append(str(source_file) if source_file else None)
            checks['captured_'+kind+'_source_file'] = bool(source_file and source_file.is_file()
                and sha256(source_file) == captured_sources[-1])
            checks['captured_'+kind+'_seed'] = measured(row,'seed') is not None
        checks['captured_reset_id'] = bool(measured(captured,'reset_id'))
        checks['captured_source_sha256'] = bool(all(captured_sources) and len(set(captured_sources)) == 1)
        checks['captured_source_file_identity'] = bool(all(source_files) and len(set(source_files)) == 1)
        source_parts = Path(source_files[0]).parts if source_files[0] else ()
        source_start = max((index for index,part in enumerate(source_parts) if part == 'json_clean'),default=-1)
        identity = str(Path(*source_parts[source_start+1:])) if source_start >= 0 else None
        checks['canonical_captured_source_identity'] = bool(identity and len(Path(identity).parts) == 5)
        # Pre-controller checkpoint results may not yet have condition_snapshot.
        # Preserve the receipt's captured source separately from the raw result.
        binding['source_sha256'] = captured_sources[0]
        declared_sources = _source_hashes(result)
        checks['result_source_consistency'] = not declared_sources or declared_sources == set(captured_sources)
    except (OSError,ValueError,TypeError,KeyError,AttributeError) as exception:
        checks['checkpoint_provenance_readable'] = False
        error = f'{type(exception).__name__}: {exception}'
    accepted = all(checks.values())
    missing = [key for key,value in checks.items() if not value]
    return {'status':'available' if accepted else 'unavailable','accepted':accepted,
        'path':str(receipt_path.resolve()) if accepted else None,'sha256':receipt_sha if accepted else None,
        'unbound_receipt_file':str(receipt_path.resolve()) if receipt_sha and not accepted else None,
        'unbound_receipt_sha256':receipt_sha if not accepted else None,
        'checks':checks,'missing_proof':missing,
        'reason':None if accepted else error or 'Checkpoint receipt context mismatch: ' + ', '.join(missing),
        'provenance_origin':'actual_initialization_camera_mounts_provenance',
        'captured_source_identity':identity if 'identity' in locals() else None,
        'captured_provenance':captured,'captured_variant_binding':binding}


def native_summary(path, result):
    """Preserve raw qualification/setup diagnostics without promoting control smoke tests."""
    checkpoint = (result.get('status') == 'stopped_at_requested_initialization_checkpoint'
        and result.get('diagnostic_checkpoint_complete') is True)
    checkpoint_receipt = initialization_checkpoint_proof(path,result) if checkpoint else None
    setups = []
    for phase,value in (('initial',result.get('initial_state',{})),('initial_condition',result.get('initial_condition',{})),
                        ('initialization_diagnostics',result.get('initialization_diagnostics',{})),
                        ('final',result.get('final_state',{})),('qualification',result.get('qualification',{}).get('checks',{}))):
        if not isinstance(value,dict):continue
        for name in ('interaction_setup','interaction_initialization'):
            receipt=value.get(name)
            if isinstance(receipt,dict):setups.append({'phase':phase,'receipt':deepcopy(receipt)})
    commands=result.get('commands',[])
    controls=[{'command':deepcopy(row.get('command')),'completed':row.get('result',{}).get('completed'),
               'reason':row.get('result',{}).get('reason')}
              for row in commands if isinstance(row,dict)] if isinstance(commands,list) else []
    qualification=result.get('qualification',{})
    final=result.get('final_state',{})
    loaded_start=(final.get('loaded_room_start') or {}) if isinstance(final,dict) else {}
    initial_status=('accepted' if loaded_start.get('accepted') is True else 'observed_rejected'
                    if loaded_start.get('accepted') is False else 'unknown')
    # Prefer the final measured setup, while retaining every initial/rejected attempt.
    final_setups=[row for row in setups if row['phase']=='final']
    latest_setup=(final_setups or setups)[-1]['receipt'] if setups else {}
    layout_status=('accepted' if latest_setup.get('accepted') is True else 'observed_rejected'
                   if latest_setup.get('accepted') is False else 'unknown')
    # Only explicit autonomous measurement can set this; complete diagnostic and
    # successful public controls never establish Agent/task/grasp performance.
    autonomous=bool(result.get('autonomous_agent_performance') is True and result.get('model_calls',0)>0
                    and result.get('agent_performance_measurement'))
    initial_dir=path.parent/'initial'
    mounted={name:str((initial_dir/(name+'.png')).resolve()) for name in MOUNTED if (initial_dir/(name+'.png')).is_file()}
    global_image=initial_dir/'initial_overview_rgb.png'
    captures={'initial_global':str(global_image.resolve()) if global_image.is_file() else None,
        'initial_mounted':mounted,'all_eight_mounted_present':len(mounted)==8,
        'initial_capture_status':'available' if global_image.is_file() and len(mounted)==8 else 'pending',
        'other_review_images':[str(p.resolve()) for p in sorted(path.parent.glob('**/*.png')) if 'human_review' in str(p)]}
    if checkpoint:
        # Stage measurements do not certify an ordinary complete episode reset,
        # even if the directory contains older or independent image artifacts.
        initial_status = layout_status = 'unknown'
        autonomous = False
        captures['initial_capture_status'] = 'pending'
    return {'evidence_id':path.parent.name,'private_result_file':str(path.resolve()),'result_sha256':sha256(path),
        'evidence_record_type':'native_checker_result',
        'diagnostic_complete':False if checkpoint else result.get('complete'),
        'diagnostic_status':'requested_initialization_checkpoint' if checkpoint else deepcopy(result.get('status')),
        'diagnostic_checkpoint_complete':result.get('diagnostic_checkpoint_complete'),
        'checkpoint_stage':result.get('checkpoint_stage'),
        'benchmark_execution_verified':False if checkpoint else result.get('benchmark_execution_verified'),
        'initialization_checkpoint_evidence':checkpoint_receipt,
        'qualification':deepcopy(qualification),
        'interaction_setup_evidence':setups,'loaded_room_start':deepcopy(loaded_start),
        'public_control_results':controls,'public_control_success_count':sum(row['completed'] is True for row in controls) if controls else None,
        'native_initialization_status':initial_status,'layout_status':layout_status,
        'execution_status':'initialization_checkpoint_only' if checkpoint else 'public_control_only' if controls else 'diagnostic_only',
        'scoring_status':'unavailable','autonomous_agent_performance':autonomous,
        'agent_task_success':None,'grasp_success':None,'benchmark_task_success':None,
        'raw_diagnostic_fields':{key:deepcopy(result.get(key)) for key in ('status','failure_stage','failure_reason',
            'error','traceback','public_controls_only','model_calls','elapsed_s','diagnostic_only','benchmark_world_mode',
            'truth_available','native_protocol_error','inventory_reset','calibration_receipt',
            'initialization_diagnostics','diagnostic_overrides','diagnostic_checkpoint_complete',
            'checkpoint_stage','benchmark_execution_verified') if key in result},
        'captures':captures}


def runtime_abort_proof(path, result):
    """Bind an external stop record to its same-run captured construction context.

    Companion files are read only through the three explicit abort paths. Their
    existence proves retained diagnostics, not the claimed error's causal asset.
    Current variant matching remains a separate mandatory check.
    """
    directory = path.parent.resolve()
    checks = {'schema':result.get('schema') == 'native_runtime_abort_v1',
        'evidence_origin':result.get('evidence_origin') == ABORT_ORIGIN,
        'incomplete_diagnostic':result.get('complete') is False and result.get('diagnostic_only') is True,
        'no_agent_measurement':result.get('model_calls') == 0
            and result.get('autonomous_agent_performance') is False,
        'no_execution_certificate':result.get('benchmark_execution_verified') is False
            and result.get('capture_or_task_success_claimed') is False}
    files = {}
    for key in ('native_error_log','native_stack_file','native_construction_receipt'):
        filename = absolute(result.get(key), directory)
        file = Path(filename) if filename else None
        valid = bool(file and file.is_relative_to(directory) and file.is_file())
        checks[key + '_same_run_file'] = valid
        files[key] = {'path':filename,'sha256':sha256(file) if valid else None}
        if result.get(key + '_sha256'):
            checks[key + '_declared_sha256'] = files[key]['sha256'] == result[key + '_sha256']
    construction = {}
    captured = {}
    try:
        if not checks['native_construction_receipt_same_run_file']:
            raise ValueError('Same-run native construction receipt is missing.')
        construction = json.loads(Path(files['native_construction_receipt']['path']).read_text())
        cameras = construction['camera_mounts']
        checks['native_constructed_stage'] = (construction.get('stage') == 'constructed'
            and bool(construction.get('source_constructor'))
            and cameras.get('schema') == 'initial_camera_mount_diagnostic_v1'
            and cameras.get('stage') == 'constructed')
        provenance = cameras['provenance']
        captured = deepcopy(provenance)
        def measured(row, key):
            value = row.get(key, {})
            return value.get('value') if isinstance(value,dict) and value.get('available') is True else None
        source_paths = []
        for kind in ('task','scene'):
            row = provenance.get(kind, {})
            for key, field in (('file',kind + '_file'),('file_sha256',kind + '_sha256'),
                               ('source_sha256','source_sha256'),('source_file','source_file')):
                value = measured(row,key)
                expected = result.get(field)
                checks['captured_' + kind + '_' + key] = bool(value and expected and
                    (absolute(value) == absolute(expected) if key in ('file','source_file') else value == expected))
            checks['captured_' + kind + '_seed'] = measured(row,'seed') is not None
            source_paths.append(measured(row,'source_file'))
        checks['captured_reset_id'] = bool(measured(provenance,'reset_id'))
        revision = result.get('frozen_revision')
        checks['captured_frozen_revision'] = bool(isinstance(revision,str) and revision and all(
            filename and revision in Path(filename).parts for filename in source_paths))
    except (OSError,ValueError,TypeError,KeyError,AttributeError) as error:
        checks['construction_provenance_readable'] = False
        files['construction_error'] = f'{type(error).__name__}: {error}'
    return {'accepted':all(checks.values()),'checks':checks,
        'missing_proof':[key for key,value in checks.items() if not value],
        'artifacts':files,'captured_construction_provenance':captured}


def runtime_abort_summary(path, result, proof):
    """An observed abort is neither a finalized checker nor benchmark execution."""
    return {'evidence_id':path.parent.name + '_runtime_abort',
        'evidence_record_type':'externally_recorded_runtime_abort',
        'evidence_origin':result.get('evidence_origin'),
        'private_result_file':str(path.resolve()),'private_abort_file':str(path.resolve()),
        'result_sha256':sha256(path),'abort_sha256':sha256(path),
        'diagnostic_complete':False,'checker_finalized_diagnostic':False,
        'diagnostic_status':'observed_native_runtime_error' if proof['accepted'] else 'unbound_native_runtime_abort',
        'native_runtime_status':'observed_native_runtime_error' if proof['accepted'] else 'unbound_native_runtime_abort',
        'abort_binding_proof':proof,'qualification':None,'interaction_setup_evidence':[],
        'loaded_room_start':None,'public_control_results':[],'public_control_success_count':None,
        'native_initialization_status':'unknown','layout_status':'unknown',
        'execution_status':'unavailable','scoring_status':'unavailable',
        'autonomous_agent_performance':False,'agent_task_success':None,'grasp_success':None,
        'benchmark_task_success':None,'task_success':None,'score_percent':None,'completion_score':None,
        'raw_diagnostic_fields':deepcopy(result),
        'captures':{'initial_global':None,'initial_mounted':{},'all_eight_mounted_present':False,
            'initial_capture_status':'pending','other_review_images':[]}}


def build_manifest(baseline_path=BASELINE, *, root=ROOT, input_root=None, source_root=None, output=None):
    root=Path(root).resolve();input_root=Path(input_root or root/'outputs/task_edit2_20261018').resolve()
    source_root=Path(source_root or root/'ESI/dataset/json_clean').resolve()
    baseline_path=Path(baseline_path).resolve();baseline=json.loads(baseline_path.read_text())
    originals=baseline['tasks']
    identities=[row['source'] for row in originals]
    if len(identities)!=len(set(identities)):
        raise ValueError('Original Core membership must preserve one unique row per source.')
    builder=Builder(root,source_root)
    variants={};sources={};original_keys=set()
    for original in originals:
        record={**deepcopy(original),'source_identity':original['source']}
        variant=builder.variant(record,str(baseline_path),original=True)
        variants[variant['variant_key']]=variant;original_keys.add(variant['variant_key'])
        source=variant['sources'][0]
        row={'source_identity':original['source'],'source_file':source['source_file'],'source_sha256':source['source_sha256'],
            'original_candidate_id':original['candidate_id'],'original_task_file':original.get('task_file'),
            'original_scene_file':original.get('scene_file'),'original_variant_key':variant['variant_key'],
            'original_membership_record':deepcopy(original),'derivative_memberships':[],
            'coverage_status':'original_preserved_no_explicit_derivative','final_selection_owner':'user'}
        paths=[]
        job=(original.get('capture') or {}).get('job_id')
        if job:
            directory=baseline_path.parent/'captures/jobs'/job
            paths=[str(p.resolve()) for p in sorted(directory.glob('attempts/*/*.png'))]
        row['historical_review_images']={'association':'original_manifest_job_id_only_not_current_qualification','paths':paths}
        image=input_root/'original_esi_images'/Path(original['source']).with_suffix('.png')
        row['original_esi_images']=[str(image.resolve())] if image.is_file() else []
        sources[row['source_identity']]=row
    inputs=[];missing=[]
    for relative in CURRENT_INPUTS+OPTIONAL_INPUTS:
        path=input_root/relative
        if not path.is_file():
            missing.append({'path':str(path),'optional':relative in OPTIONAL_INPUTS});continue
        data=json.loads(path.read_text());inputs.append({'path':str(path),'sha256':sha256(path),'schema':data.get('schema')})
        records=data.get('records',data.get('candidates'))
        if not isinstance(records,list):
            builder.ambiguities.append({'manifest':str(path),'reason':'explicit_records_or_candidates_list_missing'});continue
        for record in records:
            variant=builder.variant(record,str(path))
            existing=variants.get(variant['variant_key'])
            if existing:
                existing['manifest_records'].extend(row for row in variant['manifest_records'] if row not in existing['manifest_records'])
                merged={row['source_identity']:deepcopy(row) for row in existing['sources']}
                for member in variant['sources']:
                    previous=merged.setdefault(member['source_identity'],deepcopy(member))
                    for field in ('declared_source_hashes','mappings'):
                        previous[field].extend(value for value in member[field] if value not in previous[field])
                existing['sources']=[builder.source(member) for member in merged.values()]
                binding_states={member['provenance_status'] for member in existing['sources']}
                declared_sources=set(merged)
                if 'hash_mismatch' in binding_states:
                    existing['source_binding_status']='source_hash_mismatch'
                elif 'unavailable' in binding_states:
                    existing['source_binding_status']='unavailable'
                elif (existing['source_binding_status'] != 'verified_declared_protected_parent_linkage'
                      and declared_sources and all((existing[kind].get('source') or {}).get('json') in declared_sources for kind in ('task','scene'))):
                    existing['source_binding_status']='verified'
                existing['pending_gates']=sorted(set(existing['pending_gates']+variant['pending_gates']))
                continue
            variants[variant['variant_key']]=variant
    additions={}
    for key,variant in variants.items():
        if key in original_keys:continue
        core_members=[member['source_identity'] for member in variant['sources'] if member['source_identity'] in sources]
        extra_members=[member['source_identity'] for member in variant['sources'] if member['source_identity'] not in sources]
        variant['original_core_members']=core_members;variant['additional_source_members']=extra_members
        variant['membership_class']='mixed_core_and_additional_members' if core_members and extra_members else 'core_derivative' if core_members else 'additional_prototype'
        for member in variant['sources']:
            identity=member['source_identity']
            reference={'variant_key':key,'candidate_id':variant['candidate_id'],'member_mappings':member['mappings']}
            if identity in sources:
                sources[identity]['derivative_memberships'].append(reference)
                sources[identity]['coverage_status']='explicit_derivative_available_for_review'
            else:
                row=additions.setdefault(identity,{'source_identity':identity,'source_file':member['source_file'],
                    'source_sha256':member['source_sha256'],'original_core_member':False,'variant_memberships':[],
                    'selection_status':'addition_only_user_review'})
                row['variant_memberships'].append(reference)
    evidence=[];unmatched=[]
    native_paths = sorted([*input_root.glob('native_*/result.json'), *input_root.glob('native_*/runtime_abort.json')])
    for path in native_paths:
        is_abort = path.name == 'runtime_abort.json'
        try:
            result=json.loads(path.read_text())
            if not isinstance(result,dict):raise ValueError('Native result must be a JSON object.')
        except (OSError,ValueError) as error:
            unmatched.append({'result_file':str(path),'reason':'unreadable_result','error':str(error)});continue
        proof = runtime_abort_proof(path,result) if is_abort else None
        summary = runtime_abort_summary(path,result,proof) if is_abort else native_summary(path,result)
        if is_abort and not proof['accepted']:
            unmatched.append({'result_file':str(path.resolve()),'reason':'unbound_runtime_abort_missing_native_proof',
                'missing_proof':proof['missing_proof'],'diagnostic_complete':False,'unbound_diagnostic':summary})
            continue
        matches=[];rejections=[]
        for variant in variants.values():
            matched,checks=exact_native_match(variant,result,root,
                checkpoint_proof=summary.get('initialization_checkpoint_evidence'))
            if matched:matches.append(variant)
            elif checks['task_path'] or checks['scene_path']:
                rejections.append({'variant_key':variant['variant_key'],'checks':checks})
        if not matches:
            unmatched.append({'result_file':str(path.resolve()),'reason':'no_exact_explicit_variant_task_scene_source_hash_match',
                'declared_task_file':result.get('task_file'),'declared_scene_file':result.get('scene_file'),
                'diagnostic_complete':False if is_abort else result.get('complete'),
                'error':result.get('error'),'rejections':rejections,
                'missing_proof':['current_variant_exact_task_scene_source_binding'] if is_abort else [],
                'unbound_diagnostic':summary})
            continue
        evidence.append(summary)
        for variant in matches:
            variant['native_evidence_ids'].append(summary['evidence_id'])
    by_evidence={row['evidence_id']:row for row in evidence}
    for variant in variants.values():
        receipts=[by_evidence[key] for key in variant['native_evidence_ids']]
        variant['observed_native_statuses']={}
        for field in ('native_initialization_status','layout_status','execution_status','scoring_status','native_runtime_status'):
            statuses=sorted({row.get(field,'unavailable') for row in receipts})
            variant['observed_native_statuses'][field]=statuses
            measured=[value for value in statuses if value not in ('unknown','unavailable')]
            if measured:
                variant[field]=measured[0] if len(measured)==1 else 'mixed_native_attempts'
        variant['autonomous_agent_performance']=any(row['autonomous_agent_performance'] for row in receipts)
        variant['artifacts']['current_initial_capture_status']=('available' if any(
            row['captures']['initial_capture_status']=='available' for row in receipts) else 'pending')
        variant['pending_review_gates']=list(variant['pending_gates'])
        for label,field in (('native_initialization','native_initialization_status'),('native_layout','layout_status')):
            if variant[field]!='accepted':variant['pending_review_gates'].append(label)
        if not variant['autonomous_agent_performance']:
            variant['pending_review_gates'].append('autonomous_agent_execution_unmeasured')
        if variant['scoring_status']=='unavailable':variant['pending_review_gates'].append('terminal_scoring_unmeasured')
        if variant['artifacts']['current_initial_capture_status']=='pending':
            variant['pending_review_gates'].append('current_initial_global_and_eight_mounted_captures')
        variant['pending_review_gates']=sorted(set(variant['pending_review_gates']))
    counts={'original_core_sources':len(sources),'original_baseline_variants':len(original_keys),
        'explicit_derivative_variants':len(variants)-len(original_keys),
        'original_sources_with_explicit_derivatives':sum(bool(row['derivative_memberships']) for row in sources.values()),
        'original_sources_without_explicit_derivatives':sum(not row['derivative_memberships'] for row in sources.values()),
        'additional_sources':len(additions),'additional_only_variants':sum(v.get('membership_class')=='additional_prototype' for v in variants.values()),
        'mixed_core_and_additional_variants':sum(v.get('membership_class')=='mixed_core_and_additional_members' for v in variants.values()),
        'schema_statuses':dict(Counter(v['schema_status'] for v in variants.values())),
        'source_binding_statuses':dict(Counter(v['source_binding_status'] for v in variants.values())),
        'native_initialization_statuses':dict(Counter(v['native_initialization_status'] for v in variants.values())),
        'native_layout_statuses':dict(Counter(v['layout_status'] for v in variants.values())),
        'current_initial_global_and_eight_capture_variants':sum(v['artifacts']['current_initial_capture_status']=='available' for v in variants.values()),
        'matched_native_results':sum(row['evidence_record_type']=='native_checker_result' for row in evidence),
        'matched_native_runtime_aborts':sum(row['evidence_record_type']=='externally_recorded_runtime_abort' for row in evidence),
        'matched_native_evidence':len(evidence),'unmatched_native_diagnostics':len(unmatched),
        'pending_review_gates':dict(Counter(gate for variant in variants.values() for gate in variant['pending_review_gates'])),
        'mapping_ambiguities':len(builder.ambiguities),'autonomous_agent_variants':sum(v['autonomous_agent_performance'] for v in variants.values())}
    manifest={'schema':'task_edit2_provisional_review_20261018_v1','status':'provisional_user_review',
        'final_selection_owner':'user','final_core_selection_made':False,
        'baseline_membership':{'path':str(baseline_path),'sha256':sha256(baseline_path),'source_field':'tasks/source'},
        'explicit_current_inputs':inputs,'missing_explicit_inputs':missing,
        'native_scan_policy':str(input_root/'native_*/result.json'),
        'runtime_abort_scan_policy':str(input_root/'native_*/runtime_abort.json'),'counts':counts,
        'core_sources':list(sources.values()),'variants':list(variants.values()),'additional_sources':list(additions.values()),
        'native_evidence':evidence,'unmatched_native_diagnostics':unmatched,'mapping_ambiguities':builder.ambiguities,
        'interpretation_rules':['Definition schema validity is independent of native initialization/layout/execution/scoring.',
            'A complete diagnostic or successful public control is not grasp/task/Agent success.',
            'No family-level evidence transfer; missing measurements remain unknown or unavailable.',
            'Historical review images do not qualify current task physics or mandatory Agent inputs.']}
    if output is not None:
        output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
        (output/'current_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
        (output/'report.md').write_text(report(manifest,output/'current_manifest.json'))
    return manifest


def report(manifest,path):
    c=manifest['counts']
    return ('# Provisional TaskEdit2 review coverage\n\n'
        f"All {c['original_core_sources']} original Core source rows and their historical IDs/definitions are preserved. "
        'Final selection belongs to the user; this manifest promotes no final Core variant.\n\n'
        f"- Explicit current derivative variants: {c['explicit_derivative_variants']}.\n"
        f"- Original sources with explicit derivatives: {c['original_sources_with_explicit_derivatives']}; "
        f"without a current derivative: {c['original_sources_without_explicit_derivatives']}.\n"
        f"- Additional sources: {c['additional_sources']}; additional-only variants: {c['additional_only_variants']}; "
        f"mixed Core/additional parents: {c['mixed_core_and_additional_variants']}.\n"
        f"- Definition schema statuses: {json.dumps(c['schema_statuses'],sort_keys=True)}.\n"
        f"- Exact-bound native diagnostic results: {c['matched_native_results']}; "
        f"externally recorded runtime aborts: {c['matched_native_runtime_aborts']}; "
        f"unmatched native diagnostics: {c['unmatched_native_diagnostics']}.\n"
        f"- Autonomous Agent performance variants: {c['autonomous_agent_variants']}.\n"
        f"- Unresolved manifest mappings: {c['mapping_ambiguities']}.\n\n"
        f"Current full Global/eight-mounted capture sets are exact-bound for {c['current_initial_global_and_eight_capture_variants']} "
        'variants; other capture gates remain pending. Explicit task/scene source differences are recorded '
        'for review with both source hashes, without adding scene-only sources to original Core membership.\n\n'
        'Protected original Rigid question-to-parent-scene links are classified separately only after exact '
        'declared-path and shared loader verification; this static provenance does not transfer native qualification.\n\n'
        'Definition validity, native initialization, layout, execution and scoring are separate fields. '
        'Absent evidence remains unknown/unavailable. Native skipped qualifications and failed setup attempts '
        'remain diagnostic records; complete diagnostic runs or successful controls do not establish task, '
        'grasp or Agent success. Current initial Global/eight-camera captures remain pending where missing. '
        'Native route/access/semantic/action/terminal and calibrated-budget gates remain candidate-specific.\n\n'
        f'[Complete current manifest]({path}) includes exact hashes, source membership, original records, '
        'explicit derivative choices, additions, native rejection reasons and absolute artifact links.\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=BASELINE)
    parser.add_argument('--input-root',type=Path,default=ROOT/'outputs/task_edit2_20261018')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/task_edit2_20261018/review')
    args=parser.parse_args(argv)
    manifest=build_manifest(args.baseline,input_root=args.input_root,output=args.output)
    print(json.dumps(manifest['counts'],sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
