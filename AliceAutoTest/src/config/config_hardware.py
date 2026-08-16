"""
硬件配置模块 - 跨平台实现
使用psutil统一处理，最小化平台差异
"""

import contextlib
import getpass
import os
import socket

import psutil

from src.config.base_config import GetVersion
from src.config.platform_compat import get_platform, is_windows


class HardwareConfig:
    """硬件配置类 - 跨平台实现"""

    def __init__(self):
        self.platform = get_platform()
        try:
            self._gv = GetVersion()
        except Exception:
            self._gv = None

    def get_hostname(self) -> str:
        """获取主机名"""
        return socket.gethostname()

    def get_netcard(self, name: str = "") -> list:
        """获取网卡信息"""
        netcard_info = []
        info = psutil.net_if_addrs()
        for iface, addrs in info.items():
            for addr in addrs:
                # IPv4地址且不是回环地址
                if (
                    addr.family == socket.AF_INET
                    and addr.address != "127.0.0.1"
                ):
                    netcard_info.append((iface, addr.address))

        if name:
            for iface, ip in netcard_info:
                if iface == name:
                    return [(iface, ip)]
        return netcard_info

    def get_mac(self, interface: str = "") -> str:
        """获取MAC地址"""
        if interface:
            # 获取指定网卡的MAC
            addrs = psutil.net_if_addrs()
            if interface in addrs:
                for addr in addrs[interface]:
                    if addr.family == psutil.AF_LINK:
                        return addr.address
        # 使用跨平台方法
        return self.platform.get_mac_address()

    @staticmethod
    def get_user() -> str:
        """获取当前用户名"""
        return getpass.getuser()

    def get_process_list(self) -> list:
        """获取进程列表"""
        process_list = []
        for proc in psutil.process_iter(["pid", "name"]):
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                process_list.append(
                    {"pid": proc.info["pid"], "name": proc.info["name"]}
                )
        return process_list


class ProcessManager:
    """进程管理器 - 跨平台实现"""

    def __init__(self):
        self.platform = get_platform()

    def get_pid(self, name: str) -> int:
        """根据进程名获取PID"""
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                if proc.info["name"] == name:
                    return proc.info["pid"]
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return None

    def kill(self, name: str) -> bool:
        """根据进程名终止进程"""
        killed = False
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                if proc.info["name"] == name:
                    proc.terminate()
                    proc.wait(timeout=3)
                    killed = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            except psutil.TimeoutExpired:
                try:
                    proc.kill()
                    killed = True
                except Exception:
                    pass
        return killed

    def kill_by_pid(self, pid: int) -> bool:
        """根据PID终止进程"""
        try:
            proc = psutil.Process(pid)
            proc.terminate()
            proc.wait(timeout=3)
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
        except psutil.TimeoutExpired:
            try:
                proc.kill()
                return True
            except Exception:
                return False

    def run_kill(self, names: list = None):
        """批量终止进程"""
        if names is None:
            names = ["VCamDemo", "VCamDemo.exe"]

        for name in names:
            try:
                self.kill(name)
            except Exception as e:
                print(f"终止进程 {name} 失败: {e}")


class PidConfig(ProcessManager):
    """Phoenix进程配置"""

    def __init__(self):
        super().__init__()
        try:
            self._gv = GetVersion()
        except Exception:
            self._gv = None

    def copy_mockfile(self) -> bool:
        """复制mock文件"""
        if not self._gv:
            return False
        try:
            mock_path = self._gv.get_mock()
        except Exception:
            return False
        if os.path.exists(mock_path):
            return True

        # 使用跨平台路径
        src_mock = os.path.join(
            self.platform.get_app_data_path(),
            "BaseConfig",
            "mocker_answer_config.ini",
        )

        if os.path.exists(src_mock):
            return self.platform.copy_file(src_mock, mock_path)
        return False


# Windows特定的注册表操作（仅在Windows上可用）
class WindowsRegistry:
    """Windows注册表操作"""

    @staticmethod
    def read_key(hkey, path: str, name: str):
        """读取注册表键值"""
        if not is_windows():
            return None

        try:
            import winreg

            with winreg.OpenKey(hkey, path) as key:
                value, _ = winreg.QueryValueEx(key, name)
                return value
        except Exception as e:
            print(f"读取注册表失败: {e}")
            return None

    @staticmethod
    def write_key(hkey, path: str, name: str, value, reg_type=None):
        """写入注册表键值"""
        if not is_windows():
            return False

        try:
            import winreg

            if reg_type is None:
                reg_type = winreg.REG_SZ
            with winreg.CreateKey(hkey, path) as key:
                winreg.SetValueEx(key, name, 0, reg_type, value)
            return True
        except Exception as e:
            print(f"写入注册表失败: {e}")
            return False


if __name__ == "__main__":
    hw = HardwareConfig()
    print(f"当前用户: {hw.get_user()}")
    print(f"主机名: {hw.get_hostname()}")
    print(f"MAC地址: {hw.get_mac()}")
    print(f"IP地址: {hw.platform.get_ip_address()}")

    pm = ProcessManager()
    print(f"进程数量: {len(hw.get_process_list())}")
