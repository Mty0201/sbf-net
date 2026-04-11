from __future__ import annotations

import sys
from pathlib import Path


def ensure_zaha_root_on_path() -> None:
    """Allow direct execution of data_pre/zaha/scripts/ without PYTHONPATH setup."""
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
