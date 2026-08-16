#!/usr/bin/env python3
"""FantasyAI 无第三方依赖的目录与 Python 语法检查。"""

import ast
import json
import subprocess
import sys
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGNORED_TOP_LEVEL = {"docs", "tools", "__pycache__"}
ARTIFACT_PARTS = {"build", "checkpoints", "log", "logs", "output", "outputs", "runs"}
ARTIFACT_SUFFIXES = {".engine", ".h5", ".onnx", ".pt", ".pth"}
PROJECT_INSTRUCTION_FILES = {"AGENTS.md", "CLAUDE.md"}


def subprojects():
    return sorted(
        path
        for path in ROOT.iterdir()
        if path.is_dir() and path.name not in IGNORED_TOP_LEVEL and not path.name.startswith(".")
    )


def tracked_files():
    try:
        result = subprocess.run(
            ["git", "ls-files", ROOT.name],
            cwd=ROOT.parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    repository_root = ROOT.parent
    return [repository_root / line for line in result.stdout.splitlines() if line]


def main():
    errors = []
    warnings_found = []
    projects = subprojects()

    for project in projects:
        if not (project / "README.md").is_file():
            errors.append(f"{project.relative_to(ROOT)}: 缺少根 README.md")

    for markdown in ROOT.rglob("*.md"):
        relative = markdown.relative_to(ROOT)
        if (
            markdown.name.lower() == "readme.md"
            or markdown.name in PROJECT_INSTRUCTION_FILES
            or "docs" in relative.parts
        ):
            continue
        errors.append(f"{relative}: 非 README 文档应放入 docs/")

    python_count = 0
    for source in ROOT.rglob("*.py"):
        if "__pycache__" in source.parts:
            continue
        python_count += 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
        except (OSError, SyntaxError, UnicodeError) as exc:
            errors.append(f"{source.relative_to(ROOT)}: {exc}")

    notebook_count = 0
    for notebook in ROOT.rglob("*.ipynb"):
        if ".ipynb_checkpoints" in notebook.parts:
            continue
        notebook_count += 1
        try:
            payload = json.loads(notebook.read_text(encoding="utf-8"))
            if not isinstance(payload.get("cells"), list) or "nbformat" not in payload:
                raise ValueError("缺少 cells 或 nbformat")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{notebook.relative_to(ROOT)}: Notebook 格式错误: {exc}")

    for path in tracked_files():
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            continue
        if (
            ARTIFACT_PARTS.intersection(relative.parts)
            or path.suffix.lower() in ARTIFACT_SUFFIXES
            or path.name.startswith("events.out.tfevents.")
        ):
            warnings_found.append(str(relative))

    print(
        f"FantasyAI structure: {len(projects)} subprojects, "
        f"{python_count} Python files, {notebook_count} notebooks"
    )
    if warnings_found:
        print(f"WARN: {len(warnings_found)} tracked model/runtime artifacts (see docs/subprojects.md)")
    for error in errors:
        print(f"FAIL: {error}")
    if errors:
        print(f"Result: FAIL ({len(errors)} errors)")
        return 1
    print("Result: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
