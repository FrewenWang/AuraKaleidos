#!/usr/bin/env python3
"""Validate the declared structure of first-party projects in this monorepo."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "config/projects.json"
GENERATED_PARTS = {"__pycache__", "node_modules", ".gradle", ".idea", ".pytest_cache"}


def tracked_large_files(limit_mib: int) -> list[tuple[int, str]]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    findings = []
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        path = ROOT / relative
        if path.is_file() and (size := path.stat().st_size) > limit_mib * 1024 * 1024:
            findings.append((size, relative))
    return sorted(findings, reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Treat large tracked files as errors")
    parser.add_argument("--large-file-mib", type=int, default=50)
    args = parser.parse_args()

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    projects = catalog["projects"]
    errors: list[str] = []
    seen: set[str] = set()

    for project in projects:
        project_path = project["path"]
        if project_path in seen:
            errors.append(f"duplicate project entry: {project_path}")
        seen.add(project_path)
        root = ROOT / project_path
        if not root.is_dir():
            errors.append(f"missing project directory: {project_path}")
            continue
        for required in project["required"]:
            if not (root / required).exists():
                errors.append(f"{project_path}: missing {required}")
        package_json = root / "package.json"
        if package_json.is_file():
            package = json.loads(package_json.read_text(encoding="utf-8"))
            if main_entry := package.get("main"):
                if not (root / main_entry).is_file():
                    errors.append(f"{project_path}: package main does not exist: {main_entry}")

    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        path = raw_path.decode("utf-8", errors="surrogateescape")
        if GENERATED_PARTS.intersection(Path(path).parts):
            errors.append(f"generated path is tracked: {path}")

    large_files = tracked_large_files(args.large_file_mib)
    for size, path in large_files:
        message = f"tracked large file ({size / 1024 / 1024:.1f} MiB): {path}"
        if args.strict:
            errors.append(message)
        else:
            print(f"WARNING: {message}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(f"Validated {len(projects)} first-party project layouts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
