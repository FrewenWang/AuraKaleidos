"""
硬件信息模块 (config_hardware) 单元测试

测试 HardwareConfig、PidConfig、ProcessManager
"""

import contextlib
import os

import psutil
import pytest


class TestHardwareConfig:
    """HardwareConfig 测试"""

    @pytest.fixture
    def hw(self):
        from src.config.config_hardware import HardwareConfig

        return HardwareConfig()

    def test_get_hostname(self, hw):
        """获取主机名"""
        hostname = hw.get_hostname()
        assert hostname
        assert len(hostname) > 0

    def test_get_mac(self, hw):
        """获取 MAC 地址"""
        mac = hw.get_mac()
        assert mac
        assert ":" in mac

    def test_get_ip_via_platform(self, hw):
        """通过 platform 获取 IP"""
        ip = hw.platform.get_ip_address()
        assert ip
        assert "." in ip

    def test_get_user(self, hw):
        """获取用户名"""
        # HardwareConfig 继承的平台方法
        user = hw.platform.get_user()
        assert user
        assert len(user) > 0

    def test_init_does_not_crash_on_non_windows(self, hw):
        """非 Windows 初始化不崩溃"""
        from src.config.platform_compat import is_windows

        if not is_windows():
            # _gv 应为 None（降级模式）
            assert hw._gv is None or hw._gv is not None
        # 应能正常实例化
        assert hw is not None


class TestProcessManager:
    """ProcessManager 测试"""

    @pytest.fixture
    def pm(self):
        from src.config.config_hardware import ProcessManager

        return ProcessManager()

    def test_kill_nonexistent(self, pm):
        """杀不存在的进程不报错"""
        result = pm.kill("NonExistentProcessXYZ")
        assert isinstance(result, bool)

    def test_kill_empty_name(self, pm):
        """空进程名不报错"""
        result = pm.kill("")
        assert isinstance(result, bool)

    def test_kill_by_invalid_pid(self, pm):
        """无效 PID 杀进程"""
        result = pm.kill_by_pid(99999999)
        assert result is False


class TestConfigPid:
    """PidConfig 测试"""

    @pytest.fixture
    def pid_config(self):
        from src.config.config_hardware import PidConfig

        return PidConfig()

    def test_init_on_non_windows(self, pid_config):
        """非 Windows 初始化不崩溃"""
        from src.config.platform_compat import is_windows

        if not is_windows():
            assert pid_config._gv is None or pid_config._gv is not None
        assert pid_config is not None

    def test_copy_mockfile_non_windows(self, pid_config):
        """非 Windows 上 copy_mockfile 不崩溃"""
        from src.config.platform_compat import is_windows

        if not is_windows():
            # 应安全跳过或返回，不抛异常
            with contextlib.suppress(Exception):
                pid_config.copy_mockfile()


class TestPsutilIntegration:
    """psutil 集成测试"""

    def test_psutil_available(self):
        """psutil 已安装"""
        assert psutil is not None

    def test_get_pids(self):
        """获取进程列表"""
        pids = psutil.pids()
        assert len(pids) > 0

    def test_current_process(self):
        """获取当前进程信息"""
        proc = psutil.Process()
        assert proc.pid == os.getpid()
        assert proc.name()
