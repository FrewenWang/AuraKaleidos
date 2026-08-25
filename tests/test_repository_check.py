import json
import tempfile
import unittest
from pathlib import Path

from tools.repository_check import (
    credential_matches,
    local_markdown_targets,
    lockfile_root_mismatches,
    notebook_source_text,
    run_checks,
    windows_path_issue,
)


class MarkdownLinkTest(unittest.TestCase):
    def test_only_local_targets_are_returned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs = root / "docs"
            docs.mkdir()
            markdown = docs / "index.md"
            markdown.write_text(
                "[本地](guide.md) [锚点](#usage) "
                "[网络](https://example.com) [根路径](/README.md)\n"
                "```cpp\ncallback([](const Value& value) {});\n```",
                encoding="utf-8",
            )

            targets = local_markdown_targets(markdown, root)

        self.assertEqual(
            targets,
            [
                (docs / "guide.md").resolve(strict=False),
                (root / "README.md").resolve(strict=False),
            ],
        )


class WindowsPathTest(unittest.TestCase):
    def test_rejects_reserved_and_invalid_names(self):
        self.assertTrue(windows_path_issue("docs/CON.txt"))
        self.assertTrue(windows_path_issue("docs/bad?.txt"))
        self.assertTrue(windows_path_issue("docs/trailing."))

    def test_accepts_portable_path(self):
        self.assertFalse(windows_path_issue("docs/工程维护与测试指南.md"))


class NodeLockfileTest(unittest.TestCase):
    def test_detects_stale_root_dependencies(self):
        package = {"dependencies": {"cheerio": "1.0.0"}}
        lockfile = {
            "lockfileVersion": 3,
            "packages": {"": {"dependencies": {"cheerio": "^0.22.0"}}},
        }

        self.assertEqual(lockfile_root_mismatches(package, lockfile), ["dependencies"])

    def test_accepts_old_lockfile_without_root_package_metadata(self):
        self.assertEqual(lockfile_root_mismatches({}, {"lockfileVersion": 1}), [])


class NotebookCredentialTest(unittest.TestCase):
    def test_scans_inputs_but_ignores_notebook_outputs(self):
        fake_key = "sk-" + ("1" * 32)
        with tempfile.TemporaryDirectory() as directory:
            notebook = Path(directory) / "example.ipynb"
            notebook.write_text(
                json.dumps(
                    {
                        "cells": [
                            {
                                "cell_type": "code",
                                "source": [
                                    "import os\n",
                                    "api_key = os.environ['DEEPSEEK_API_KEY']\n",
                                ],
                                "outputs": [
                                    {
                                        "output_type": "stream",
                                        "text": fake_key,
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            content = notebook_source_text(notebook)

        self.assertEqual(credential_matches(content), [])

    def test_detects_hard_coded_api_key(self):
        content = f'api_key = "{"sk-" + ("1" * 32)}"'

        self.assertEqual(credential_matches(content), [("OpenAI/DeepSeek API key", 1)])


class RepositoryIntegrationTest(unittest.TestCase):
    def test_current_repository_passes_quick_checks(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(run_checks(root), [])


if __name__ == "__main__":
    unittest.main()
