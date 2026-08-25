"""Phoenix 客户端进程生命周期的跨平台封装。"""

import subprocess
from pathlib import Path

from src.config.config_hardware import ProcessManager
from src.config.platform_compat import is_windows

PHOENIX_PROCESS_NAMES = (
    "Phoenix.exe",
    "Phoenix",
    "VCamDemo.exe",
    "VCamDemo",
)


def cleanup_phoenix_processes(process_manager=None):
    """清理已知 Phoenix/VCam 进程并返回逐项结果。"""
    manager = process_manager or ProcessManager()
    return {name: manager.kill(name) for name in PHOENIX_PROCESS_NAMES}


def start_phoenix_client(project_root=None, windows=None, runner=None):
    """仅在 Windows 上从项目内批处理脚本启动 Phoenix 客户端。"""
    windows = is_windows() if windows is None else windows
    if not windows:
        return False

    root = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    start_script = root / "config" / "start_phoenix.bat"
    if not start_script.is_file():
        return False

    popen = runner or subprocess.Popen
    popen(
        ["cmd.exe", "/c", str(start_script)],
        cwd=str(start_script.parent),
    )
    return True
