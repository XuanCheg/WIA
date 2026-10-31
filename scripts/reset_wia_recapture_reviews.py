"""Reset selected recapture decisions through the live review API, retaining notes.

Run with behavior Python and PYTHONPATH=src:
  python scripts/reset_wia_recapture_reviews.py --aliases T22 D1-1 --apply
Without --apply, list the selected rows only. Existing records are archived by
the review service; no feedback file is rewritten outside that service.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from wia.resources import ROOT


def request(base, route, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(base + route, data=data, headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aliases', nargs='+', required=True)
    parser.add_argument('--base-url', default='http://127.0.0.1:8626')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    snapshot = ROOT.parent / 'Show-Harness-Explore/outputs/wia_v1_20261023/review/snapshot.json'
    rows = {r['document_alias']: r for r in json.loads(snapshot.read_text())['rows']}
    aliases = list(dict.fromkeys(args.aliases))
    missing = set(aliases) - rows.keys()
    if missing:
        raise ValueError('Unknown selected aliases: ' + ', '.join(sorted(missing)))
    if not args.apply:
        print(json.dumps({'aliases': aliases, 'mode': 'dry_run'}))
        return
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'outputs/review_repairs_20261030/review_resets' / run_id
    output.mkdir(parents=True)
    results = []
    for alias in aliases:
        row = rows[alias]
        variant = row['variants'][0]['id']
        query = urlencode({'id': row['id'], 'variant_id': variant})
        before = request(args.base_url, '/api/scene?' + query)['scene']
        prior = before['review']
        payload = {'variant_id': variant, 'decision': 'unreviewed',
                   'notes': prior.get('notes', ''), 'revision': prior.get('revision', 0),
                   'fingerprint': before['fingerprint']}
        (output / (alias + '.before.json')).write_text(json.dumps(prior, ensure_ascii=False, indent=2) + '\n')
        result = request(args.base_url, '/api/reviews/' + row['id'], payload)['review']
        after = request(args.base_url, '/api/scene?' + query)['scene']
        if (result['decision'] != 'unreviewed' or after['decision'] != 'unreviewed'
                or result['notes'] != payload['notes'] or after['review']['notes'] != payload['notes']):
            raise RuntimeError('Review reset verification failed for ' + alias)
        results.append({'alias': alias, 'row_id': row['id'], 'variant_id': variant,
                        'previous_decision': prior['decision'], 'decision': result['decision'],
                        'notes_preserved': True, 'revision': result['revision']})
        (output / 'result.json').write_text(json.dumps({'rows': results}, indent=2) + '\n')
    print(json.dumps({'reset_count': len(results), 'all_notes_preserved': True, 'report': str(output / 'result.json')}))


if __name__ == '__main__':
    main()
