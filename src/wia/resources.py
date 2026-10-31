"""Resolve checkout resources. Run: python -c 'from wia.resources import ROOT; print(ROOT)'."""
import os
from pathlib import Path

ROOT = Path(os.environ.get('WIA_ROOT', Path(__file__).resolve().parents[2])).expanduser().resolve()
DATA_ROOT = ROOT / 'explore_to_achieve'
ESI_ROOT = Path(os.environ.get('WIA_ESI_ROOT', ROOT / 'ESI')).expanduser().resolve()
TASK_ROOT = DATA_ROOT / 'wia_v1/release/tasks'
SCENE_ROOT = DATA_ROOT / 'scenes'
UPSTREAM_ROOT = ESI_ROOT / 'dataset/json_clean'
AUTHORED_ROOT = DATA_ROOT / 'sources/json_clean'
DATASET_ROOT = Path(os.environ.get('WIA_DATASET_ROOT') or ROOT).expanduser().resolve()
CATALOGUE_PATH = (DATASET_ROOT / 'catalogue.json' if os.environ.get('WIA_DATASET_ROOT')
                  else ROOT / 'explore_to_achieve/wia_v1/release/catalogue.json')



def resolve_resource_path(path, root=ROOT):
    """Resolve explicit current resources without historical relocation or hash fallbacks."""
    value = str(path)
    if value.startswith('dataset://'):
        return _within(DATASET_ROOT, value[len('dataset://'):])
    if value.startswith('behavior://'):
        data = os.environ.get('OMNIGIBSON_DATA_PATH')
        if not data:
            raise ValueError('OMNIGIBSON_DATA_PATH is required for licensed native scene resources.')
        return _within(Path(data).expanduser(), value[len('behavior://'):])
    path = Path(path).expanduser()
    return (Path(root) / path).resolve()


def _within(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Resource identity must stay inside its declared root.')
    return path
