#!/usr/bin/env python3
"""对 AuraKaleidos 聚合仓库执行快速、无外部副作用的结构检查。"""

from __future__ import annotations

import argparse
import configparser
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

TOP_LEVEL_PROJECTS = (
    "AliceAndroid",
    "AliceAutoTest",
    "AliceJava",
    "FantasyAI",
    "FantasyAIAgent",
    "FantasyAlgorithm",
    "FantasyAutoDrive",
    "FantasyCXX",
    "FantasyCuda",
    "FantasyFlutter",
    "FantasyHPC",
    "FantasyJS",
    "FantasyKotlin",
    "FantasyNodeJS",
    "FantasyPython",
    "FantasyShell",
    "FantasySwiftIOS",
    "FantasyToolkits",
)

WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
MARKDOWN_LINK = re.compile(r"!?\[[^]]*]\(([^)]+)\)")
NOTEBOOK_CREDENTIAL_PATTERNS = (
    ("OpenAI/DeepSeek API key", re.compile(r"\bsk-[A-Za-z0-9]{24,}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "GitHub token",
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b", flags=re.IGNORECASE),
    ),
    (
        "DingTalk access token",
        re.compile(r"access_token=[A-Za-z0-9_-]{20,}", flags=re.IGNORECASE),
    ),
)


def maintained_markdown_files(root: Path) -> list[Path]:
    """返回由本仓库直接维护、应保证本地链接有效的 Markdown 文件。"""
    files = [root / "README.md"]
    files.extend(root / project / "README.md" for project in TOP_LEVEL_PROJECTS)
    files.extend(sorted((root / "docs").glob("*.md")))
    for project in TOP_LEVEL_PROJECTS:
        files.extend(sorted((root / project / "docs").glob("*.md")))
    return [path for path in files if path.is_file()]


def local_markdown_targets(markdown: Path, root: Path) -> list[Path]:
    """解析 Markdown 中的本地链接目标；网络地址和页内锚点不返回。"""
    targets: list[Path] = []
    content = markdown.read_text(encoding="utf-8")
    content = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
    content = re.sub(r"`[^`\n]*`", "", content)
    for raw_target in MARKDOWN_LINK.findall(content):
        target = raw_target.strip()
        if target.startswith("<") and ">" in target:
            target = target[1 : target.index(">")]
        elif " " in target:
            target = target.split(" ", 1)[0]
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        link_path = Path(unquote(parsed.path))
        if link_path.is_absolute():
            destination = root / str(link_path).lstrip("/\\")
        else:
            destination = markdown.parent / link_path
        targets.append(destination.resolve(strict=False))
    return targets


def check_project_readmes(root: Path) -> list[str]:
    issues = []
    for project in TOP_LEVEL_PROJECTS:
        directory = root / project
        if not directory.is_dir():
            issues.append(f"缺少一级工程目录：{project}")
        elif not (directory / "README.md").is_file():
            issues.append(f"缺少一级工程说明：{project}/README.md")
    return issues


def check_markdown_links(root: Path) -> list[str]:
    issues = []
    for markdown in maintained_markdown_files(root):
        for target in local_markdown_targets(markdown, root):
            if not target.exists():
                source = markdown.relative_to(root)
                try:
                    missing = target.relative_to(root)
                except ValueError:
                    missing = target
                issues.append(f"失效文档链接：{source} -> {missing}")
    return issues


def node_manifests(root: Path) -> list[Path]:
    manifests = [root / "FantasyJS" / "package.json"]
    manifests.extend((root / "FantasyNodeJS").glob("*/package.json"))
    manifests.extend((root / "FantasyNodeJS").glob("*/*/package.json"))
    return sorted(path for path in manifests if path.is_file())


def lockfile_root_mismatches(package: dict, lockfile: dict) -> list[str]:
    """比较 npm lockfile v2/v3 的根依赖声明与 package.json。"""
    locked_root = lockfile.get("packages", {}).get("")
    if not isinstance(locked_root, dict):
        return []
    mismatches = []
    for group in ("dependencies", "devDependencies", "optionalDependencies"):
        expected = package.get(group, {})
        actual = locked_root.get(group, {})
        if expected != actual:
            mismatches.append(group)
    return mismatches


def check_node_manifests(root: Path) -> list[str]:
    issues = []
    for manifest in node_manifests(root):
        relative = manifest.relative_to(root)
        try:
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            issues.append(f"无效 JSON：{relative}（{error}）")
            continue
        if not metadata.get("name"):
            issues.append(f"Node 工程缺少 name：{relative}")
        if not metadata.get("scripts", {}).get("test"):
            issues.append(f"Node 工程缺少 test 脚本：{relative}")
        lock_path = manifest.with_name("package-lock.json")
        if not lock_path.is_file():
            issues.append(f"Node 工程缺少 package-lock.json：{relative.parent}")
            continue
        try:
            lockfile = json.loads(lock_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            issues.append(f"无效 JSON：{lock_path.relative_to(root)}（{error}）")
            continue
        mismatches = lockfile_root_mismatches(metadata, lockfile)
        if mismatches:
            issues.append(
                f"Node lockfile 与 package.json 不同步：{relative.parent} "
                f"({', '.join(mismatches)})"
            )
    return issues


def tracked_entries(root: Path) -> list[tuple[str, str]]:
    """返回 Git 索引中的 ``(mode, path)``；没有 Git 时抛出可读错误。"""
    process = subprocess.run(
        ["git", "ls-files", "-s", "-z"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    entries = []
    for record in process.stdout.decode("utf-8").split("\0"):
        if not record:
            continue
        metadata, path = record.split("\t", 1)
        mode = metadata.split(" ", 1)[0]
        entries.append((mode, path))
    return entries


def notebook_source_text(notebook: Path) -> str:
    """只提取 Notebook 代码/Markdown 单元，避免输出中的 base64 造成误报。"""
    document = json.loads(notebook.read_text(encoding="utf-8"))
    fragments = []
    for cell in document.get("cells", []):
        source = cell.get("source", [])
        if isinstance(source, str):
            fragments.append(source)
        elif isinstance(source, list):
            fragments.append("".join(str(line) for line in source))
    return "\n".join(fragments)


def credential_matches(content: str) -> list[tuple[str, int]]:
    """返回文本中疑似硬编码凭据的类型和行号。"""
    matches = []
    for label, pattern in NOTEBOOK_CREDENTIAL_PATTERNS:
        for match in pattern.finditer(content):
            line = content.count("\n", 0, match.start()) + 1
            matches.append((label, line))
    return matches


def check_notebook_credentials(root: Path) -> list[str]:
    """扫描已跟踪 Notebook 的输入单元，禁止提交看似真实的固定凭据。"""
    try:
        entries = tracked_entries(root)
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        UnicodeDecodeError,
    ) as error:
        return [f"无法扫描 Notebook 凭据：{error}"]

    issues = []
    for mode, relative in entries:
        if mode == "160000" or not relative.endswith(".ipynb"):
            continue
        notebook = root / relative
        if not notebook.is_file():
            continue
        try:
            content = notebook_source_text(notebook)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            issues.append(f"无法解析 Notebook：{relative}（{error}）")
            continue
        for label, line in credential_matches(content):
            issues.append(f"Notebook 疑似硬编码凭据：{relative}:{line}（{label}）")
    return issues


def windows_path_issue(path: str) -> bool:
    """判断 Git 路径是否无法在常见 Windows 文件系统中检出。"""
    for component in Path(path).parts:
        stem = component.split(".", 1)[0].upper()
        if (
            any(character in component for character in '<>:"\\|?*')
            or component.endswith((" ", "."))
            or stem in WINDOWS_RESERVED_NAMES
        ):
            return True
    return False


def check_git_portability(root: Path) -> list[str]:
    issues = []
    try:
        entries = tracked_entries(root)
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        UnicodeDecodeError,
    ) as error:
        return [f"无法读取 Git 索引：{error}"]

    casefolded: dict[str, list[str]] = {}
    gitlinks = set()
    tracked = set()
    for mode, relative in entries:
        tracked.add(relative)
        casefolded.setdefault(relative.casefold(), []).append(relative)
        if windows_path_issue(relative):
            issues.append(f"Windows 不兼容路径：{relative}")
        path = root / relative
        if mode == "120000" and path.is_symlink() and not path.exists():
            issues.append(f"失效符号链接：{relative} -> {path.readlink()}")
        if mode == "160000":
            gitlinks.add(relative)

    for variants in casefolded.values():
        if len(variants) > 1:
            issues.append(f"大小写冲突路径：{' | '.join(variants)}")

    sensitive_paths = {
        "AliceAutoTest/config/config.ini",
        "AliceAutoTest/config/config.local.ini",
        "AliceAutoTest/.env",
    }
    for relative in sorted(sensitive_paths & tracked):
        issues.append(f"本地敏感配置不应纳入 Git：{relative}")

    modules = configparser.ConfigParser()
    modules.read(root / ".gitmodules", encoding="utf-8")
    declared_gitlinks = {
        modules.get(section, "path")
        for section in modules.sections()
        if modules.has_option(section, "path")
    }
    for relative in sorted(gitlinks - declared_gitlinks):
        issues.append(f"Git 子模块缺少 .gitmodules 声明：{relative}")
    for relative in sorted(declared_gitlinks - gitlinks):
        issues.append(f".gitmodules 声明未对应 Git 子模块：{relative}")
    return issues


def run_checks(root: Path) -> list[str]:
    checks = (
        check_project_readmes,
        check_markdown_links,
        check_node_manifests,
        check_git_portability,
        check_notebook_credentials,
    )
    return [issue for check in checks for issue in check(root)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="仓库根目录（默认根据脚本位置推导）",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    issues = run_checks(root)
    if issues:
        print(f"仓库快速检查失败，共 {len(issues)} 项：", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 1
    print(
        f"仓库快速检查通过：{len(TOP_LEVEL_PROJECTS)} 个一级工程，"
        f"{len(maintained_markdown_files(root))} 份维护文档，"
        f"{len(node_manifests(root))} 个 Node 工程。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
