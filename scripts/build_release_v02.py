"""Build three portable v0.2 source archives without modifying live experiments.

Run: python scripts/build_release_v02.py --output-dir dist/v0.2
Requires PyYAML. Output must not exist. No network, GPU or model calls are made.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs/releases/v0.2'
VERSION = '0.2.0'
ESI_COMMIT = '9f6f15ed43e81654a97de9e4db2a1fedadca2bc1'
PORTABLE_TESTS = ['test_usage.py', 'test_candidate_events.py', 'test_candidate_high.py',
                  'test_candidate_medium.py', 'test_candidate_inventory.py']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    target.chmod(0o755 if os.access(source, os.X_OK) else 0o644)


def files(root, directories):
    """Allowlist tracked or visible source files, excluding ignored local material."""
    output = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z',
        '--cached', '--others', '--exclude-standard', '--', *directories])
    for value in sorted(set(output.decode().split('\0')) - {''}):
        path = root / value
        if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts:
            yield Path(value)


def portable_metadata(value):
    """Relocate provenance labels, keeping them distinct from runtime resources."""
    if isinstance(value, dict):
        return {k: portable_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable_metadata(v) for v in value]
    if not isinstance(value, str):
        return value
    if value.startswith('/'):
        if '/data/behavior/' in value:
            return 'behavior://' + value.split('/data/behavior/', 1)[1]
        for project in ('WIA', 'Show-Harness-Explore', 'InteractiveWorld'):
            marker = '/' + project + '/'
            if marker in value:
                return 'provenance://' + project + '/' + value.split(marker, 1)[1]
        if value.startswith(('/home/', '/local_home', '/tmp/', '/mnt/', '/root/')):
            raise ValueError('Unclassified machine-specific dataset path: ' + value)
        # JSON pointers and native USD prim paths are identities, not local files.
        return value
    return value


def strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, str):
        yield value


def dataset(destination):
    original = json.loads((ROOT / 'explore_to_achieve/wia_v1/release/catalogue.json').read_text())
    rows, source_refs, transformations = [], {}, []
    for row in original['tasks']:
        result = {k: deepcopy(row[k]) for k in ('alias', 'task_id', 'variant', 'family', 'subclass', 'origin') if k in row}
        for key, folder in [('task_file', 'tasks'), ('scene_file', 'scenes')]:
            source = ROOT / row[key]
            value = yaml.safe_load(source.read_text())
            # Keep the schema-required empty native baseline, plus the selected initialization.
            value['scene']['variants'] = {k: value['scene']['variants'][k]
                                          for k in dict.fromkeys(('original', row['variant']))}
            for identity in strings(value):
                candidate = Path(identity)
                if not candidate.is_absolute() and len(candidate.parts) == 5 and candidate.suffix == '.json':
                    upstream = ROOT / 'ESI/dataset/json_clean' / candidate
                    authored = ROOT / 'explore_to_achieve/sources/json_clean' / candidate
                    if upstream.is_file() or authored.is_file():
                        source_refs[identity] = upstream if upstream.is_file() else authored
            value = portable_metadata(value)
            target = Path(folder) / (row['alias'].lower() + '.yaml')
            (destination / target).parent.mkdir(parents=True, exist_ok=True)
            (destination / target).write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True, width=100))
            result[key] = str(target)
            result[key + '_sha256'] = digest(destination / target)
            transformations.append({'published': str(target), 'original': row[key],
                                    'original_sha256': digest(source), 'published_sha256': digest(destination / target)})
            if key == 'task_file':
                result['title'] = row.get('title', value['public']['objective'])
        rows.append(result)
    assert len(rows) == 60 and len({r['alias'] for r in rows}) == 60
    dependencies = []
    for identity, source in sorted(source_refs.items()):
        entry = {'identity': identity, 'original_sha256': digest(source)}
        if source.is_relative_to(ROOT / 'ESI'):
            entry.update(location='external_esi', path='dataset/json_clean/' + identity,
                         sha256=digest(source))
        else:
            target = destination / 'sources/json_clean' / identity
            raw = json.loads(source.read_text())
            cleaned = portable_metadata(raw)
            if raw == cleaned:
                copy(source, target)
            else:
                write_json(target, cleaned)
            entry.update(location='bundled_authored_source', path=str(target.relative_to(destination)),
                         sha256=digest(target))
        dependencies.append(entry)
    catalogue = {'schema': 'wia_dataset_release_v2', 'version': VERSION,
        'selection': 'main_60', 'selection_revision': original['selection_revision'],
        'qualification': 'selected_definitions_not_universal_physical_certification',
        'active_family_counts': dict(sorted(Counter(r['family'] for r in rows).items())),
        'taxonomy': portable_metadata(original['taxonomy']), 'tasks': rows,
        'excluded_policy': 'Extension selections, retired D tasks and redundant candidate variants are not distributed.'}
    write_json(destination / 'catalogue.json', catalogue)
    write_json(destination / 'provenance/definitions.json', {'transformations': transformations,
        'changes': ['Keep the selected variant and schema-required empty original baseline.', 'Relocate metadata paths and the licensed T07 scene reference.',
                    'YAML formatting changes; goals, scoring, numerical settings and task IDs are preserved.']})
    write_json(destination / 'dependencies.json', {'esi_repository': 'https://github.com/ESI-Bench/ESI-Bench',
        'esi_commit': ESI_COMMIT, 'sources': dependencies,
        'licensed_native_assets': 'Install separately; OMNIGIBSON_DATA_PATH points to the BEHAVIOR data directory.'})
    with (destination / 'tasks.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['alias', 'task_id', 'family', 'subclass', 'origin', 'variant', 'task_file', 'scene_file'])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in rows)
    summary = json.loads((ROOT / 'explore_to_achieve/wia_v1/balance_20261031/household_validation.json').read_text())
    write_json(destination / 'validation/household_static_summary.json', portable_metadata(summary))
    followup = ROOT / 'explore_to_achieve/wia_v1/review_followup_20261031'
    for path in sorted(followup.glob('*.json')):
        write_json(destination / 'validation' / path.name, portable_metadata(json.loads(path.read_text())))
    protocol_review(destination, evaluation_component=False)
    for name in ['DATASET_CARD.md', 'KNOWN_ISSUES.md', 'CHANGELOG.md', 'THIRD_PARTY.md', 'VALIDATION.md']:
        copy(DOCS / name, destination / name)
    copy(DOCS / 'DATASET_README.md', destination / 'README.md')
    copy(ROOT / 'LICENSE', destination / 'LICENSE')
    copy(DOCS / 'NOTICE', destination / 'NOTICE')
    return rows


def protocol_review(destination, *, evaluation_component):
    """Include the human audit with links that work in sibling suite components."""
    code_prefix = '' if evaluation_component else '../wia-evaluation-v0.2/'
    text = (DOCS / 'PROTOCOL_REVIEW.zh-CN.md').read_text()
    text = text.replace(str(DOCS) + '/', '')
    text = text.replace(str(ROOT) + '/src/', code_prefix + 'src/')
    text = text.replace(str(ROOT) + '/explore_to_achieve/', code_prefix + 'explore_to_achieve/')
    (destination / 'PROTOCOL_REVIEW.zh-CN.md').write_text(text)
    for path in sorted((DOCS / 'protocol_review').iterdir()):
        if path.suffix == '.json':
            write_json(destination / 'protocol_review' / path.name, portable_metadata(json.loads(path.read_text())))
        else:
            copy(path, destination / 'protocol_review' / path.name)


def evaluation(destination):
    command_root = ROOT / 'src/wia/runtime/commands'
    command_names = {p.name for p in command_root.glob('*.py')}
    needed = {'__init__.py', 'run_esi_explore_episode.py', 'run_esi_robot.py', 'run_esi_agent.py',
              'run_gumi_gateway.py', 'review_wia.py', 'submit_esi_capture.py', 'run_esi_capture_worker.py'}
    # Keep subprocess/import dependencies of runtime utilities, not dated authoring entry points.
    def references(path):
        tree = ast.parse(path.read_text())
        docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                      if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                      and node.body and isinstance(node.body[0], ast.Expr)
                      and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)}
        found = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                found.update(set(re.findall(r'[a-zA-Z0-9_]+\.py', node.value)) & command_names)
            if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('wia.runtime.commands.'):
                found.add(node.module.rsplit('.', 1)[-1] + '.py')
            if isinstance(node, ast.Import):
                found.update(n.name.rsplit('.', 1)[-1] + '.py' for n in node.names
                             if n.name.startswith('wia.runtime.commands.'))
        return found
    for path in (ROOT / 'src/wia').rglob('*.py'):
        if 'commands' not in path.parts:
            needed.update(references(path))
    examined = set()
    while needed - examined:
        name = sorted(needed - examined)[0]
        examined.add(name)
        if (command_root / name).is_file():
            needed.update(references(command_root / name))
    for relative in files(ROOT, ['src/wia', 'assets', 'explore_to_achieve/prompts']):
        if relative.is_relative_to('src/wia/authoring/commands') and relative.name != '__init__.py':
            continue
        if relative.is_relative_to('src/wia/runtime/commands') and relative.name not in needed:
            continue
        if relative == Path('src/wia/diagnostics/model_worker.py'):
            continue
        if relative.suffix in {'.py', '.json', '.yaml', '.yml', '.md', '.txt', '.html', '.css', '.js', '.usda', '.svg'}:
            copy(ROOT / relative, destination / relative)
    for relative in files(ROOT, ['configs']):
        if relative.name.startswith('esi_') and relative.suffix == '.json' or relative.parent.name == 'esi_explore':
            copy(ROOT / relative, destination / relative)
    for name in ['pyproject.toml', 'LICENSE']:
        copy(ROOT / name, destination / name)
    for name in PORTABLE_TESTS:
        copy(ROOT / 'tests' / name, destination / 'tests' / name)
    copy(ROOT / 'tests/harness_integration/test_rigid_insertion_scoring.py', destination / 'tests/test_rigid_insertion_scoring.py')
    copy(ROOT / 'scripts/verify_release_v02.py', destination / 'scripts/verify_release_v02.py')
    copy(ROOT / 'tests/test_external_dataset.py', destination / 'tests/test_external_dataset.py')
    copy(DOCS / 'EVALUATION_README.md', destination / 'README.md')
    protocol_review(destination, evaluation_component=True)
    for name in ['KNOWN_ISSUES.md', 'CHANGELOG.md', 'THIRD_PARTY.md', 'NOTICE', 'VALIDATION.md']:
        copy(DOCS / name, destination / name)
    for name in ['one_task.yaml', 'suite.yaml']:
        copy(DOCS / name, destination / 'examples' / name)


def harness(source, destination):
    directories = ['core', 'showharness', 'gumi', 'plugins', 'interpreters', 'configs',
                   'prompts', 'requirements', 'scripts', 'train', 'assets', 'models/chat_templates']
    for relative in files(source, directories):
        if relative.is_relative_to('assets/robolab_franka'):
            continue
        if relative.suffix in {'.py', '.json', '.yaml', '.yml', '.md', '.txt', '.html', '.css', '.js', '.cjs',
                               '.sh', '.usda', '.svg', '.png', '.jpg', '.example', '.jinja'}:
            copy(source / relative, destination / relative)
    for name in ['pyproject.toml', 'LICENSE', 'CITATION.cff']:
        copy(source / name, destination / name)
    for name in ['finetuned.md', 'franka.md', 'piper.md', 'mcp_server.md', 'simulators.md']:
        copy(source / 'docs' / name, destination / 'docs' / name)
    for name in ['__init__.py', 'test_dataset_interface.py', 'test_config_overlays.py', 'test_execution_token_swap.py']:
        copy(source / 'tests' / name, destination / 'tests' / name)
    # The component keeps its own 0.1.0 API version; this is the v0.2 suite snapshot.
    copy(DOCS / 'HARNESS_README.md', destination / 'README.md')
    copy(DOCS / 'THIRD_PARTY.md', destination / 'THIRD_PARTY.md')
    copy(DOCS / 'NOTICE', destination / 'NOTICE')
    copy(DOCS / 'VALIDATION.md', destination / 'VALIDATION.md')


def seal(directory, base_commit):
    entries = {str(p.relative_to(directory)): {'sha256': digest(p), 'bytes': p.stat().st_size}
               for p in sorted(directory.rglob('*')) if p.is_file()}
    write_json(directory / 'MANIFEST.json', {'suite_version': VERSION, 'component': directory.name,
        'base_commit': base_commit, 'source_state': 'working_tree_snapshot', 'files': entries})
    lines = [digest(p) + '  ' + str(p.relative_to(directory)) for p in sorted(directory.rglob('*')) if p.is_file()]
    (directory / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')


def archive(directory):
    path = Path(str(directory) + '.tar.gz')
    # Fixed metadata makes repeated exports of the same source bytes identical.
    with path.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w') as tar:
            for item in sorted(directory.rglob('*')):
                info = tar.gettarinfo(str(item), arcname=str(Path(directory.name) / item.relative_to(directory)))
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ''
                if item.is_file():
                    with item.open('rb') as stream:
                        tar.addfile(info, stream)
                else:
                    tar.addfile(info)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--harness-root', type=Path, default=ROOT.parent / 'Show-Harness-Explore')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    parts = [out / 'wia-dataset-v0.2', out / 'wia-evaluation-v0.2', out / 'ShowHarnessExplore-v0.2']
    for part in parts:
        part.mkdir()
    dataset(parts[0])
    evaluation(parts[1])
    harness(args.harness_root, parts[2])
    for part, repo in zip(parts, [ROOT, ROOT, args.harness_root]):
        commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        seal(part, commit)
    archives = [archive(part) for part in parts]
    (out / 'SHA256SUMS').write_text(''.join(digest(p) + '  ' + p.name + '\n' for p in archives))
    copy(DOCS / 'README.md', out / 'README.md')
    print(json.dumps({'output': str(out), 'archives': [{'name': p.name, 'bytes': p.stat().st_size} for p in archives]}, indent=2))


if __name__ == '__main__':
    main()
