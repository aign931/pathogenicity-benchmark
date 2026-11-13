"""Resolve data and outputs within an explicit analysis workspace."""
import os
from pathlib import Path

def workspace_path(relative):
    root = Path(os.environ.get('PB_WORKSPACE', str(Path(__file__).resolve().parent))).expanduser().resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError('Path escapes PB_WORKSPACE: ' + relative)
    return str(path)
