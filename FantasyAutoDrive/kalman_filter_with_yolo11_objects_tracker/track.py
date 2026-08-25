#!/usr/bin/env python3
"""Source-checkout wrapper for the packaged object-tracker CLI."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from app.cli import run_tracking  # noqa: E402


if __name__ == "__main__":
    run_tracking()
