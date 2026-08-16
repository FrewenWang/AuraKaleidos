"""Make the local ``src`` package importable for direct script execution."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"


def add_source_root() -> None:
    source = str(SOURCE_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)
