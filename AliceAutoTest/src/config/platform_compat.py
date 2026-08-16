"""
跨平台兼容模块 - 提供统一的接口支持Windows、Mac和Linux系统
尽量减少平台特定代码，优先使用跨平台的库（如psutil、pathlib）
"""

import getpass
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

# 尝试导入可选的跨平台库
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("警告: psutil未安装，部分功能将受限")


class PlatformCompat:
    """跨平台兼容性工具类 - 最小化平台差异"""

    def __init__(self):
        self.system = platform.system().lower()
        self.is_windows = self.system == "windows"
        self.is_mac = self.system == "darwin"
        self.is_linux = self.system == "linux"

    # ==================== 路径管理 ====================

    def get_user_home(self) -> str:
        """获取用户主目录"""
        return str(Path.home())

    def get_app_data_path(self, app_name: str = "Phoenix") -> str:
        """获取应用数据目录"""
        if self.is_windows:
            base = os.environ.get(
                "APPDATA", str(Path.home() / "AppData" / "Roaming")
            )
        elif self.is_mac:
            base = str(Path.home() / "Library" / "Application Support")
        else:
            base = os.environ.get(
                "XDG_CONFIG_HOME", str(Path.home() / ".config")
            )
        return str(Path(base) / app_name)

    def get_temp_path(self) -> str:
        """获取临时文件目录"""
        return tempfile.gettempdir()

    def get_hostname(self) -> str:
        """获取主机名"""
        return socket.gethostname()

    def get_ip_address(self) -> str:
        """获取IP地址"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"

    def get_mac_address(self) -> str:
        """获取MAC地址"""
        try:
            import uuid

            mac = uuid.getnode()
            return ":".join(f"{(mac >> i) & 0xFF:02x}" for i in range(0, 48, 8))
        except Exception:
            return ""

    # ==================== 进程管理 ====================

    def kill_process(self, process_name: str) -> bool:
        """跨平台杀进程 - 优先使用psutil"""
        if not PSUTIL_AVAILABLE:
            return self._kill_process_fallback(process_name)

        killed = False
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                if (
                    proc.info["name"]
                    and process_name.lower() in proc.info["name"].lower()
                ):
                    proc.terminate()
                    proc.wait(timeout=3)
                    killed = True
            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
                psutil.TimeoutExpired,
            ):
                try:
                    proc.kill()
                    killed = True
                except Exception:
                    pass
        return killed

    def kill_process_by_pid(self, pid: int) -> bool:
        """跨平台根据PID杀进程"""
        if PSUTIL_AVAILABLE:
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                proc.wait(timeout=3)
                return True
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                try:
                    proc.kill()
                    return True
                except Exception:
                    return False
            except psutil.AccessDenied:
                return False

        # 降级方案
        try:
            if self.is_windows:
                os.system(f"taskkill /F /PID {pid}")
            else:
                os.kill(pid, signal.SIGTERM)
            return True
        except Exception:
            return False

    def _kill_process_fallback(self, process_name: str) -> bool:
        """降级方案：使用系统命令"""
        try:
            if self.is_windows:
                result = subprocess.run(
                    ["taskkill", "/F", "/IM", process_name],
                    capture_output=True,
                    text=True,
                )
            else:
                result = subprocess.run(
                    ["pkill", "-f", process_name],
                    capture_output=True,
                    text=True,
                )
            return result.returncode == 0
        except Exception:
            return False

    # ==================== 文件操作 ====================

    def ensure_dir(self, path: str) -> bool:
        """确保目录存在"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"创建目录失败 {path}: {e}")
            return False

    def copy_file(self, src: str, dst: str) -> bool:
        """复制文件"""
        try:
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            print(f"复制文件失败: {e}")
            return False

    def open_file(self, filepath: str) -> bool:
        """使用系统默认程序打开文件"""
        try:
            if self.is_windows:
                os.startfile(filepath)
            elif self.is_mac:
                subprocess.run(["open", filepath], check=False)
            else:
                subprocess.run(["xdg-open", filepath], check=False)
            return True
        except Exception as e:
            print(f"打开文件失败: {e}")
            return False

    # ==================== 命令执行 ====================

    def run_command(self, command, shell: bool = False, **kwargs) -> dict:
        """跨平台执行命令"""
        try:
            if isinstance(command, str) and not shell:
                command = command.split()

            result = subprocess.run(
                command, shell=shell, capture_output=True, text=True, **kwargs
            )

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except Exception as e:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
            }

    # ==================== 系统信息 ====================

    def get_user(self) -> str:
        """获取当前用户名"""
        return getpass.getuser()

    def is_admin(self) -> bool:
        """检查是否有管理员权限"""
        try:
            if self.is_windows:
                import ctypes

                return ctypes.windll.shell32.IsUserAnAdmin()
            else:
                return os.geteuid() == 0
        except Exception:
            return False

    # ==================== 环境检测 ====================

    def is_wsl(self) -> bool:
        """检测是否在WSL环境下"""
        if not self.is_linux:
            return False
        try:
            with open("/proc/version") as f:
                version_info = f.read().lower()
            return "microsoft" in version_info or "wsl" in version_info
        except Exception:
            return False

    # ==================== 兼容性检查 ====================

    def check_compatibility(self) -> dict:
        """检查当前环境的兼容性"""
        issues = []

        # 检查Python版本

        # 检查psutil
        if not PSUTIL_AVAILABLE:
            issues.append("psutil未安装，进程管理功能将受限")

        return {
            "compatible": len(issues) == 0,
            "issues": issues,
            "platform": self.system,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        }


# 全局单例
_platform = None


def get_platform() -> PlatformCompat:
    """获取平台兼容实例（单例模式）"""
    global _platform
    if _platform is None:
        _platform = PlatformCompat()
    return _platform


def is_windows() -> bool:
    """是否为Windows系统"""
    return get_platform().is_windows


def is_mac() -> bool:
    """是否为macOS系统"""
    return get_platform().is_mac


# 向后兼容别名
is_macos = is_mac


def is_linux() -> bool:
    """是否为Linux系统"""
    return get_platform().is_linux


# 便捷函数
def kill_process(name: str) -> bool:
    """便捷函数：杀进程"""
    return get_platform().kill_process(name)


def ensure_dir(path: str) -> bool:
    """便捷函数：确保目录存在"""
    return get_platform().ensure_dir(path)


def run_command(cmd, shell: bool = False, **kwargs) -> dict:
    """便捷函数：执行命令"""
    return get_platform().run_command(cmd, shell=shell, **kwargs)


if __name__ == "__main__":
    # 测试
    p = get_platform()
    print(f"操作系统: {p.system}")
    print(f"用户主目录: {p.get_user_home()}")
    print(f"应用数据目录: {p.get_app_data_path()}")
    print(f"临时目录: {p.get_temp_path()}")
    print(f"主机名: {p.get_hostname()}")
    print(f"IP地址: {p.get_ip_address()}")
    print(f"MAC地址: {p.get_mac_address()}")
    print(f"当前用户: {p.get_user()}")
    print(f"管理员权限: {p.is_admin()}")
    print(f"兼容性检查: {p.check_compatibility()}")
