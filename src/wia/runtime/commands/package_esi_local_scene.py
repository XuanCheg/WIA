#!/usr/bin/env python3
"""Package one native scene export and its exact local asset dependencies.

Run: python src/wia/runtime/commands/package_esi_local_scene.py --export-dir NATIVE_EXPORT
 --data-root BEHAVIOR_DATA --output-dir NEW_BUNDLE --scene-id category_HEX
 --focus-object obj1
Uses only Python's standard library. No native runtime, downloads, decryption,
scene rewriting, benchmark evaluation, or archive creation is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat

RUNTIME_VERSIONS = {
    'omnigibson': '3.7.1',
    'behavior_commit': '79f0f5fb74a6063b67c6c0ce64f85fd96f2ba429',
    'isaac_sim': '4.5.0',
    'python': '3.10',
}
SUPPORTED_CLASSES = {
    ('omnigibson.objects.dataset_object', 'DatasetObject'),
    ('omnigibson.objects.primitive_object', 'PrimitiveObject'),
    ('omnigibson.robots.r1', 'R1'),
}


def _component(value, label):
    if (not isinstance(value, str) or not value or value in ('.', '..')
            or any(character in value for character in '/\\\x00')
            or any(ord(character) < 32 for character in value)
            or any(character in value for character in ':*?"<>|')
            or value.endswith((' ', '.'))):
        raise ValueError(label + ' must be a safe single path component.')
    return value


def _source_identity(value):
    if not isinstance(value, str):
        raise ValueError('The export must declare its source_identity.')
    parts = value.split('/')
    if len(parts) != 5 or not parts[-1].endswith('.json'):
        raise ValueError('source_identity must be an exact five-part source JSON identity.')
    for part in parts:
        _component(part, 'source_identity component')
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _root(path, label):
    path = Path(path).expanduser().absolute()
    if path.is_symlink():
        raise ValueError(label + ' must not be a symlink.')
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(label + ' must be an existing directory.')
    return resolved


def _regular(path, label):
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(label + ' must be a regular, nonsymlink file.')
    return path


def _dependency_files(data_root, relative, *, directory):
    """Enumerate only the declared dependency; reject every nested symlink."""
    path = data_root
    for component in Path(relative).parts:
        _component(component, 'Dependency path component')
        path = path / component
        if path.is_symlink():
            raise ValueError('Symlink asset dependency is not portable: ' + relative)
    if not path.resolve().is_relative_to(data_root):
        raise ValueError('Asset dependency escapes data_root: ' + relative)
    if not path.exists():
        raise FileNotFoundError('Required asset dependency is missing: ' + relative)
    if not directory:
        return [_regular(path, relative)]
    if not path.is_dir():
        raise ValueError('Required asset dependency is not a directory: ' + relative)
    files = []
    for current, directories, names in os.walk(path, followlinks=False):
        directories.sort()
        names.sort()
        current = Path(current)
        for name in directories + names:
            _component(name, 'Asset filename')
            entry = current / name
            if entry.is_symlink():
                raise ValueError('Symlink asset dependency is not portable: ' + str(entry.relative_to(data_root)))
        for name in names:
            files.append(_regular(current / name, str((current / name).relative_to(data_root))))
    return files


def package_scene(export_dir, data_root, output_dir, *, scene_id, focus_objects=()):
    """Copy a validated native export plus its minimal declared asset subset.

    ``data_root`` directly contains behavior-1k-assets, omnigibson-robot-assets
    and omnigibson.key. ``asset_inventory`` measures data files only. All inputs
    are validated before creating the absent output directory; interrupted copy
    failures may leave an incomplete directory without a bundle.json receipt.
    """
    if not isinstance(scene_id, str) or not re.fullmatch(r'category_[0-9a-f]+', scene_id):
        raise ValueError('scene_id must be a canonical category_HEX ID.')
    export_dir = _root(export_dir, 'export_dir')
    data_root = _root(data_root, 'data_root')
    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('output_dir must be absent; existing bundles are never overwritten.')
    output = output.resolve()
    if output.is_relative_to(export_dir) or output.is_relative_to(data_root):
        raise ValueError('output_dir must be outside both input source directories.')
    input_names = ('scene.json', 'manifest.json', 'task_input.yaml', 'scene_input.yaml')
    inputs = {name: _regular(export_dir / name, name) for name in input_names}
    manifest = json.loads(inputs['manifest.json'].read_text())
    scene = json.loads(inputs['scene.json'].read_text())
    scene_hash = _sha256(inputs['scene.json'])
    if scene_hash != manifest.get('saved_scene_sha256'):
        raise ValueError('Native scene.json hash differs from saved_scene_sha256 in its export manifest.')
    if manifest.get('scene_id') is not None and manifest['scene_id'] != scene_id:
        raise ValueError('Requested scene_id differs from the exact native export scene_id.')
    source_identity = _source_identity(manifest.get('source_identity'))
    objects = scene.get('objects_info', {}).get('init_info')
    if not isinstance(objects, dict) or not objects:
        raise ValueError('Native objects_info.init_info must be a nonempty object mapping.')
    if isinstance(focus_objects, str):
        raise ValueError('focus_objects must be a sequence of native object names.')
    focuses = list(dict.fromkeys(focus_objects))
    if any(not isinstance(name, str) or name not in objects for name in focuses):
        raise ValueError('Every named focus object must exist in native objects_info.init_info.')
    models = set()
    for name, info in objects.items():
        if not isinstance(info, dict) or (info.get('class_module'), info.get('class_name')) not in SUPPORTED_CLASSES:
            raise ValueError('Unsupported native object class for ' + str(name) + ': ' +
                             str(info.get('class_module') if isinstance(info, dict) else type(info).__name__) + '.' +
                             str(info.get('class_name') if isinstance(info, dict) else ''))
        args = info.get('args')
        if not isinstance(args, dict):
            raise ValueError('Native object initialization args are missing for ' + str(name))
        if info['class_name'] == 'DatasetObject':
            models.add((_component(args.get('category'), 'DatasetObject category'),
                        _component(args.get('model'), 'DatasetObject model')))
    try:
        scene_model = _component(scene['init_info']['args']['scene_model'], 'Native scene_model')
    except (KeyError, TypeError) as error:
        raise ValueError('The native snapshot must declare init_info.args.scene_model.') from error
    directories = ['behavior-1k-assets/metadata', 'behavior-1k-assets/systems',
                   'behavior-1k-assets/scenes/' + scene_model,
                   'omnigibson-robot-assets/models/r1', 'omnigibson-robot-assets/fonts']
    directories += ['behavior-1k-assets/objects/' + category + '/' + model for category, model in sorted(models)]
    files = ['behavior-1k-assets/VERSION', 'omnigibson.key']
    for name in ('README.md', 'VERSION'):
        path = data_root / 'omnigibson-robot-assets' / name
        if path.exists() or path.is_symlink():
            files.append('omnigibson-robot-assets/' + name)
    asset_files = {}
    for relative, is_directory in [(name, True) for name in directories] + [(name, False) for name in files]:
        for path in _dependency_files(data_root, relative, directory=is_directory):
            asset_files[path.relative_to(data_root).as_posix()] = path
    # Scene bytes and initialization/state are copied verbatim, never normalized.
    output.mkdir(parents=True, exist_ok=False)
    for source_name, destination_name in (('scene.json', 'scene.json'),
            ('manifest.json', 'original_export_manifest.json'), ('task_input.yaml', 'task_input.yaml'),
            ('scene_input.yaml', 'scene_input.yaml')):
        shutil.copy2(inputs[source_name], output / destination_name)
    inventory = []
    for relative, source in sorted(asset_files.items()):
        destination = output / 'data' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        inventory.append({'path': 'data/' + relative, 'size_bytes': destination.stat().st_size})
    bundle = {'schema': 'esi_local_scene_bundle_v1', 'scene_id': scene_id,
              'source_identity': source_identity, 'native_scene_file': 'scene.json',
              'native_scene_sha256': scene_hash, 'data_dir': 'data',
              'original_export_manifest_file': 'original_export_manifest.json',
              'task_input_file': 'task_input.yaml', 'scene_input_file': 'scene_input.yaml',
              'focus_object_names': focuses, 'runtime_versions': dict(RUNTIME_VERSIONS),
              'authoring_only': True, 'qualification': False, 'model_calls': 0,
              'asset_inventory': inventory, 'total_asset_bytes': sum(row['size_bytes'] for row in inventory),
              'unique_dataset_models': [{'category': category, 'model': model} for category, model in sorted(models)]}
    (output / 'bundle.json').write_text(json.dumps(bundle, indent=2) + '\n')
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scene-id', required=True)
    parser.add_argument('--focus-object', action='append', default=[])
    args = parser.parse_args(argv)
    result = package_scene(args.export_dir, args.data_root, args.output_dir,
                           scene_id=args.scene_id, focus_objects=args.focus_object)
    print(json.dumps({'bundle_file': str(args.output_dir.resolve() / 'bundle.json'),
                      'dataset_models': len(result['unique_dataset_models']),
                      'total_asset_bytes': result['total_asset_bytes']}))


if __name__ == '__main__':
    main()
