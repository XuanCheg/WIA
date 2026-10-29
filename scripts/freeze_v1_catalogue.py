"""Freeze the reviewed preferred variants. Run: python scripts/freeze_v1_catalogue.py --snapshot FILE.

Creates a fresh release directory; historical task/source/review artifacts are not rewritten.
"""
import argparse
import hashlib
import json
from pathlib import Path

import yaml
from wia.resources import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    evidence = ROOT / 'docs/research/world_interactive_agency_v1/review_evidence.json'
    updates = json.loads(evidence.read_text())['row_updates']
    rows = json.loads(args.snapshot.read_text())['rows']
    destination = ROOT / 'explore_to_achieve/wia_v1/release'
    destination.mkdir(exist_ok=False)
    entries, excluded = [], []
    for row in rows:
        variants = row.get('variants', [])
        preferred = updates.get(row['id'], {}).get('preferred_variant_id')
        selected = next((v for v in variants if v['id'] == preferred), variants[0] if variants else {})
        if preferred and selected.get('id') != preferred:
            raise ValueError('Missing preferred variant: ' + row['document_alias'])
        if not selected.get('task_file'):
            excluded.append({'alias': row['document_alias'], 'reason': 'No executable task definition.'})
            continue
        entry = {'alias': row['document_alias'], 'family': row['primary_family'], 'variant_id': selected['id']}
        for key, folder in [('task_file', 'tasks'), ('scene_file', 'scenes')]:
            source = ROOT / selected[key]
            original = source.read_text()
            target = destination / folder / (row['document_alias'].lower() + '.yaml')
            target.parent.mkdir(exist_ok=True)
            target.write_text(original.replace('core.esi_converters.', 'wia.authoring.converters.'))
            entry[key] = str(target.relative_to(ROOT))
            entry['source_' + key] = selected[key]
            entry['source_' + key + '_sha256'] = hashlib.sha256(original.encode()).hexdigest()
        task = yaml.safe_load((ROOT / entry['task_file']).read_text())
        entry.update(task_id=task['id'], variant='adapted' if 'adapted' in task['scene']['variants'] else 'original')
        entries.append(entry)
    result = {'schema': 'wia_v1_release_catalogue', 'selection': 'review preferred variants; first variant only when no preference exists',
              'source_snapshot': str(args.snapshot.resolve()),
              'source_snapshot_sha256': hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
              'review_evidence_sha256': hashlib.sha256(evidence.read_bytes()).hexdigest(),
              'qualification': 'Task statuses and scoring semantics unchanged; selection is not execution qualification.',
              'tasks': entries, 'excluded': excluded}
    (destination / 'catalogue.json').write_text(json.dumps(result, indent=2) + '\n')
    print(len(entries), 'tasks;', len(excluded), 'excluded')


if __name__ == '__main__':
    main()
