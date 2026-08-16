"""
版本号统计模块 - 跨平台实现
"""

from pathlib import Path

from src.config.config_hardware import HardwareConfig
from src.config.platform_compat import is_windows


class ComVersion:
    """版本号统计"""

    def __init__(self):
        self.user = HardwareConfig.get_user()

    def get_cversion(self, key):
        """获取版本号（仅Windows支持）"""
        if not is_windows():
            return None

        try:
            import winreg

            reg = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
            key_path = rf"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\{{{key}}}"
            with winreg.OpenKey(reg, key_path) as k:
                version, _ = winreg.QueryValueEx(k, "DisplayVersion")
                return version
        except Exception:
            return None

    def get_mock(self):
        """获取mock文件路径"""
        if is_windows():
            return str(
                Path.home()
                / "AppData"
                / "Roaming"
                / "phoenix"
                / "bin"
                / "mocker_answer_config.ini"
            )
        else:
            return str(
                Path.home()
                / ".config"
                / "phoenix"
                / "bin"
                / "mocker_answer_config.ini"
            )
