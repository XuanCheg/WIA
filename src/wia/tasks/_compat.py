"""Preserve historical module identity. Run: python -m pytest tests/test_extraction.py."""
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader
import sys


class _AliasLoader(Loader):
    def __init__(self, canonical):
        self.canonical = canonical

    def create_module(self, spec):
        return import_module(self.canonical)

    def exec_module(self, module):
        pass


class _AliasFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        for old, new in (('core.tasks.', 'wia.tasks.'),
                         ('core.esi_converters.', 'wia.authoring.converters.')):
            if fullname.startswith(old):
                canonical = new + fullname[len(old):]
                return spec_from_loader(fullname, _AliasLoader(canonical))
        return None


def install_aliases():
    """Resolve old submodules through the canonical import exactly once."""
    if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())
