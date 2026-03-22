from __future__ import annotations

import sys
from pathlib import Path


def ensure_bf_edge_v3_root_on_path() -> None:
    """Allow direct execution of scripts/ without external PYTHONPATH setup."""
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
