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



def resolve_resource_path(path, root=ROOT):
    """Resolve explicit current resources without historical relocation or hash fallbacks."""
    path = Path(path).expanduser()
    return (Path(root) / path).resolve()
