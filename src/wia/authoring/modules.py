"""Resolve WIA-owned converter modules. Run: python -m pytest tests/test_release_catalogue.py."""
from importlib import import_module
import re

_PATTERN = re.compile(r'wia\.authoring\.converters\.[a-z][a-z0-9_]*')


def is_owned_module(name):
    return isinstance(name, str) and _PATTERN.fullmatch(name) is not None


def canonical_module_name(name):
    if not is_owned_module(name):
        raise ValueError('Use a canonical WIA category adapter.')
    return name


def import_owned_module(name):
    return import_module(canonical_module_name(name))


def module_identity(value):
    return value
