"""Locate runtime sources across packages. Run: pytest tests/harness_integration/test_explore_motion_recovery.py."""
from pathlib import Path

from wia.resources import ROOT


def runtime_source(name, root=ROOT):
    """Resolve dataset sources in WIA and reusable sources in Harness."""
    path = Path(name)
    if path.parts[0] in {'core', 'gumi'}:
        frozen_harness = Path(root) / 'harness'
        if frozen_harness.is_dir():
            return frozen_harness / path
        from showharness.resources import ROOT as harness_root
        return harness_root / path
    return Path(root) / path


def frozen_runtime_environment(root):
    """Select the sealed package sources when launching capture workers."""
    import os
    root = Path(root)
    return {'WIA_ROOT': str(root), 'WIA_ESI_ROOT': str(root / 'ESI'),
            'SHOWHARNESS_ROOT': str(root / 'harness'),
            'PYTHONPATH': os.pathsep.join((str(root / 'src'), str(root / 'harness')))}


def harness_runtime_sources():
    """Return reusable runtime files to include in a reproducible capture snapshot."""
    from showharness.resources import ROOT as harness_root
    sources = {}
    for directory in ('core', 'showharness', 'gumi', 'plugins', 'interpreters', 'configs'):
        for path in (harness_root / directory).rglob('*'):
            if path.is_file() and path.suffix in {'.py', '.json', '.yaml', '.yml', '.html', '.js', '.md'}:
                sources['harness/' + str(path.relative_to(harness_root))] = path
    return sources


def execution_resource_environment():
    """Pass split-package roots explicitly into existing persistent terminals."""
    import os
    from wia.resources import ESI_ROOT, DATASET_ROOT
    from showharness.resources import ROOT as harness_root
    return {'WIA_ROOT': str(ROOT), 'WIA_ESI_ROOT': str(ESI_ROOT),
            'WIA_DATASET_ROOT': str(DATASET_ROOT) if os.environ.get('WIA_DATASET_ROOT') else '',
            'SHOWHARNESS_ROOT': str(harness_root),
            'PYTHONPATH': os.pathsep.join(filter(None, (str(ROOT / 'src'), str(harness_root),
                                                       os.environ.get('PYTHONPATH'))))}
