"""Resolve owned converter imports without rewriting metadata. Run: python -m pytest tests/test_extraction.py."""
from importlib import import_module
import re

_PATTERN = re.compile(r'(?:core\.esi_converters|wia\.authoring\.converters)\.[a-z][a-z0-9_]*')


def is_owned_module(name):
    return isinstance(name, str) and _PATTERN.fullmatch(name) is not None


def canonical_module_name(name):
    """Validate the converter namespace before translating its import target."""
    if not is_owned_module(name):
        raise ValueError('Use an owned local category adapter.')
    return name.replace('core.esi_converters.', 'wia.authoring.converters.', 1)


def legacy_module_name(name):
    """Keep previously emitted implementation identity strings reproducible."""
    if name == 'wia.authoring.conversion':
        return 'core.esi_category_conversion'
    return name.replace('wia.authoring.converters.', 'core.esi_converters.', 1)


def import_owned_module(name):
    return import_module(canonical_module_name(name))


def module_identity(value):
    """Compare the two owned module namespaces without changing stored values."""
    return legacy_module_name(value) if is_owned_module(value) else value
