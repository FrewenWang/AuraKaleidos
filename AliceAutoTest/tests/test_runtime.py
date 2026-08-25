"""外部进程生命周期的跨平台离线测试。"""

from pathlib import Path

from src.config.runtime import (
    PHOENIX_PROCESS_NAMES,
    cleanup_phoenix_processes,
    start_phoenix_client,
)


class FakeProcessManager:
    def __init__(self):
        self.killed = []

    def kill(self, name):
        self.killed.append(name)
        return name.endswith(".exe")


def test_cleanup_covers_windows_and_unix_process_names():
    manager = FakeProcessManager()

    result = cleanup_phoenix_processes(manager)

    assert tuple(manager.killed) == PHOENIX_PROCESS_NAMES
    assert set(result) == set(PHOENIX_PROCESS_NAMES)


def test_non_windows_never_starts_batch_file(tmp_path):
    calls = []

    assert not start_phoenix_client(
        tmp_path,
        windows=False,
        runner=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    assert calls == []


def test_windows_uses_project_relative_batch_file(tmp_path):
    script = tmp_path / "config" / "start_phoenix.bat"
    script.parent.mkdir()
    script.write_text("@echo off\n", encoding="utf-8")
    calls = []

    assert start_phoenix_client(
        tmp_path,
        windows=True,
        runner=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    args, kwargs = calls[0]
    assert args[0] == ["cmd.exe", "/c", str(script)]
    assert Path(kwargs["cwd"]) == script.parent


def test_windows_missing_batch_file_is_safe(tmp_path):
    assert not start_phoenix_client(tmp_path, windows=True)
