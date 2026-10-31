"""Override explicit saved joint velocities before native scene construction.

Run CPU checks: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python -B -m pytest -q -p no:cacheprovider tests/test_wia_initial_joint_velocity.py
Use configure_scene as a before_construction category_initialization hook.
No native runtime import, initialization hook, physics step or setter is used.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from wia.resources import resolve_resource_path

REVISION = 'wia_initial_joint_velocity_v1'


def _velocities(value, name):
    if (not isinstance(value, list)
            or any(type(item) not in (int, float) or not math.isfinite(item) for item in value)):
        raise ValueError('Saved and replacement joint_vel must be finite numeric lists: ' + name)
    return deepcopy(value)


def configure_scene(config, parameters):
    """Install a source-verified copy changing only declared joint_vel arrays."""
    expected = parameters.get('expected_scene')
    if (parameters.get('revision') != REVISION or not isinstance(expected, str) or not expected.strip()
            or not isinstance(config.get('scene'), dict)
            or config['scene'].get('scene_model') != expected):
        raise ValueError('Initial joint velocity override requires its revision and exact configured scene_model.')
    declared_path = parameters.get('source_scene_file')
    if (not isinstance(declared_path, str) or not (Path(declared_path).is_absolute() or declared_path.startswith('behavior://'))
            or Path(declared_path).suffix.lower() != '.json'):
        raise ValueError('source_scene_file must be an absolute JSON file path or behavior:// resource.')
    source_path = resolve_resource_path(declared_path)
    source_bytes = source_path.read_bytes()
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    if parameters.get('source_scene_sha256') != source_digest:
        raise ValueError('Native source scene SHA256 mismatch.')
    source = json.loads(source_bytes)
    if not isinstance(source, dict):
        raise ValueError('Native source scene must be a JSON object.')
    existing = config['scene'].get('scene_file')
    if existing is not None:
        same_source = (isinstance(existing, str) and Path(existing).resolve() == source_path)
        same_source |= isinstance(existing, dict) and existing == source
        if not same_source:
            raise ValueError('Conflicting existing scene_file cannot be replaced by a joint velocity override.')
    overrides = parameters.get('overrides')
    if not isinstance(overrides, dict):
        raise ValueError('overrides must map explicit native object names to replacement joint velocities.')
    try:
        initialization = source['objects_info']['init_info']
        registry = source['state']['registry']['object_registry']
    except (KeyError, TypeError) as error:
        raise ValueError('Native source scene requires objects_info/init_info and state/registry/object_registry.') from error
    if not isinstance(initialization, dict) or not isinstance(registry, dict):
        raise ValueError('Native initialization and state object registries must be mappings.')
    copied = deepcopy(source)
    evidence = {}
    for name, replacement in overrides.items():
        if (not isinstance(name, str) or not name.strip()
                or name not in initialization or name not in registry
                or not isinstance(initialization[name], dict) or not isinstance(registry[name], dict)):
            raise ValueError('Named override object must exist in native initialization and state registries: ' + str(name))
        if 'joint_vel' not in registry[name]:
            raise ValueError('Named native object has no saved joint_vel array: ' + name)
        old = _velocities(registry[name]['joint_vel'], name)
        new = _velocities(replacement, name)
        if len(old) != len(new):
            raise ValueError('Saved and replacement joint_vel lengths must match exactly: ' + name)
        copied['state']['registry']['object_registry'][name]['joint_vel'] = new
        evidence[name] = {'old_joint_vel': old, 'new_joint_vel': deepcopy(new),
                          'state_json_path': ['state', 'registry', 'object_registry', name, 'joint_vel']}
    # All requested source arrays are validated before changing caller config.
    config['scene']['scene_file'] = copied
    return {'revision': REVISION, 'expected_scene': expected,
        'source_scene_file': str(source_path), 'source_scene_sha256': source_digest,
        'overrides': evidence, 'initialization_only': True, 'runtime_setters_used': False,
        'native_validation_run': False, 'physical_success_verified': False}
