"""Regenerate or compare the Stage 0 WIA overlay without simulator execution.

Run: python -B scripts/build_wia_registry.py [--check] [--output PATH]


Canonical invocation: python -m wia.authoring.commands.build_wia_registry --help
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from wia.resources import ROOT

from wia.tasks.wia_registry import DEFAULT_OUTPUT, build_registry, serialize_registry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--check', action='store_true', help='Compare exact bytes without writing.')
    args = parser.parse_args(argv)
    rendered = serialize_registry(build_registry())
    output = args.output.resolve()
    if args.check:
        if not output.is_file() or output.read_text(encoding='utf-8') != rendered:
            print(f'WIA registry differs or is missing: {output}', file=sys.stderr)
            return 1
        print(f'WIA registry is current: {output}')
        return 0
    if output.exists():
        import json
        try:
            previous = json.loads(output.read_text(encoding='utf-8'))
        except (ValueError, UnicodeError) as exc:
            raise ValueError('Refusing to overwrite an unrelated non-registry file.') from exc
        if not isinstance(previous, dict) or previous.get('schema') != 'world_interactive_agency_registry_v1':
            raise ValueError('Refusing to overwrite an unrelated non-registry file.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding='utf-8')
    print(f'Wrote WIA registry: {output} (61 proposals, 48 selected definitions, 13 unbuilt, 3 legacy controls)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
