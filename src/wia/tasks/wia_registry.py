"""Read the frozen WIA-v1 catalogue. Run: python -m wia tasks."""
import json
from wia.resources import ROOT

DEFAULT_OUTPUT = ROOT / 'explore_to_achieve/wia_v1/release/catalogue.json'


def build_registry(root=ROOT):
    """The release has no CoreV2 selection or legacy controls dependency."""
    return json.loads((root / 'explore_to_achieve/wia_v1/release/catalogue.json').read_text())


def serialize_registry(value):
    return json.dumps(value, indent=2) + '\n'
