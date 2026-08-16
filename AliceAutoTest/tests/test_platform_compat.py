"""
跨平台兼容模块 (platform_compat) 单元测试

测试 platform_compat.py 的所有核心功能：
- 平台检测
- 路径管理
- 系统信息
- 进程管理
- 文件操作
- 命令执行
- 兼容性检查
"""

import os

# ============================================================
# 平台检测
# ============================================================


class TestPlatformDetection:
    """平台检测测试"""

    def test_get_platform_returns_singleton(self):
        """get_platform 返回单例"""
        from src.config.platform_compat import get_platform

        p1 = get_platform()
        p2 = get_platform()
        assert p1 is p2

    def test_system_is_known(self, platform):
        """system 属性为已知值"""
        assert platform.system in ("windows", "darwin", "linux")

    def test_is_windows(self, platform):
        """is_windows 与 system 一致"""
        assert platform.is_windows == (platform.system == "windows")

    def test_is_mac(self, platform):
        """is_mac 与 system 一致"""
        assert platform.is_mac == (platform.system == "darwin")

    def test_is_linux(self, platform):
        """is_linux 与 system 一致"""
        assert platform.is_linux == (platform.system == "linux")

    def test_only_one_platform_true(self, platform):
        """只有一个平台标志为 True"""
        flags = [platform.is_windows, platform.is_mac, platform.is_linux]
        assert sum(flags) == 1

    def test_is_windows_function(self):
        """is_windows() 函数"""
        from src.config.platform_compat import is_windows

        assert isinstance(is_windows(), bool)

    def test_is_mac_function(self):
        """is_mac() 函数"""
        from src.config.platform_compat import is_mac

        assert isinstance(is_mac(), bool)

    def test_is_macos_alias(self):
        """is_macos 是 is_mac 的别名"""
        from src.config.platform_compat import is_mac, is_macos

        assert is_mac() == is_macos()

    def test_is_linux_function(self):
        """is_linux() 函数"""
        from src.config.platform_compat import is_linux

        assert isinstance(is_linux(), bool)


# ============================================================
# 路径管理
# ============================================================


class TestPathManagement:
    """路径管理测试"""

    def test_get_user_home(self, platform):
        """用户主目录存在"""
        home = platform.get_user_home()
        assert home
        assert os.path.isdir(home)

    def test_get_app_data_path_default(self, platform):
        """应用数据目录（默认 Phoenix）"""
        path = platform.get_app_data_path()
        assert "Phoenix" in path

    def test_get_app_data_path_custom(self, platform):
        """应用数据目录（自定义应用名）"""
        path = platform.get_app_data_path("WebSocketApp")
        assert "WebSocketApp" in path

    def test_get_temp_path(self, platform):
        """临时目录存在"""
        temp = platform.get_temp_path()
        assert temp
        assert os.path.isdir(temp)

    def test_get_app_data_path_format(self, platform):
        """路径格式符合当前平台"""
        path = platform.get_app_data_path()
        if platform.is_windows:
            # Windows 路径应包含盘符
            assert len(path) > 1 and path[1] == ":"
        else:
            # Unix 路径应以 / 开头
            assert path.startswith("/")


# ============================================================
# 系统信息
# ============================================================


class TestSystemInfo:
    """系统信息测试"""

    def test_get_hostname(self, platform):
        """主机名非空"""
        hostname = platform.get_hostname()
        assert hostname
        assert len(hostname) > 0

    def test_get_ip_address(self, platform):
        """IP 地址非空"""
        ip = platform.get_ip_address()
        assert ip
        # 应为有效 IP 格式或回环地址
        assert "." in ip

    def test_get_mac_address(self, platform):
        """MAC 地址格式正确"""
        mac = platform.get_mac_address()
        assert mac
        # MAC 地址应包含冒号分隔符
        assert ":" in mac

    def test_get_user(self, platform):
        """用户名非空"""
        user = platform.get_user()
        assert user
        assert len(user) > 0

    def test_is_admin_returns_bool(self, platform):
        """is_admin 返回布尔值"""
        assert isinstance(platform.is_admin(), bool)


# ============================================================
# 文件操作
# ============================================================


class TestFileOperations:
    """文件操作测试"""

    def test_ensure_dir(self, platform, temp_dir):
        """创建多级目录"""
        test_path = str(temp_dir / "a" / "b" / "c")
        assert platform.ensure_dir(test_path)
        assert os.path.isdir(test_path)

    def test_ensure_dir_idempotent(self, platform, temp_dir):
        """重复创建目录不报错"""
        test_path = str(temp_dir / "existing")
        assert platform.ensure_dir(test_path)
        assert platform.ensure_dir(test_path)  # 第二次不应报错

    def test_copy_file(self, platform, temp_dir):
        """复制文件"""
        src = temp_dir / "src.txt"
        dst = temp_dir / "dst.txt"
        src.write_text("hello world")
        assert platform.copy_file(str(src), str(dst))
        assert dst.exists()
        assert dst.read_text() == "hello world"

    def test_copy_file_nonexistent(self, platform, temp_dir):
        """复制不存在的文件返回 False"""
        src = str(temp_dir / "nonexistent.txt")
        dst = str(temp_dir / "dst.txt")
        # 应返回 False 或引发异常被捕获
        result = platform.copy_file(src, dst)
        assert result is False

    def test_open_file(self, platform, temp_dir):
        """open_file 不报错"""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test")
        # open_file 返回 bool，不验证是否真的打开
        result = platform.open_file(str(test_file))
        assert isinstance(result, bool)


# ============================================================
# 命令执行
# ============================================================


class TestCommandExecution:
    """命令执行测试"""

    def test_run_command_success(self, platform):
        """成功执行命令"""
        result = platform.run_command("echo pytest_test", shell=True)
        assert result["success"]
        assert result["returncode"] == 0
        assert "pytest_test" in result["stdout"]

    def test_run_command_failure(self, platform):
        """失败命令返回非零退出码"""
        if platform.is_windows:
            result = platform.run_command("exit 1", shell=True)
        else:
            result = platform.run_command("false", shell=True)
        assert not result["success"]
        assert result["returncode"] != 0

    def test_run_command_returns_dict(self, platform):
        """返回结果包含所有字段"""
        result = platform.run_command("echo test", shell=True)
        assert "success" in result
        assert "returncode" in result
        assert "stdout" in result
        assert "stderr" in result


# ============================================================
# 进程管理
# ============================================================


class TestProcessManagement:
    """进程管理测试"""

    def test_kill_nonexistent_process(self, platform):
        """杀不存在的进程不报错"""
        result = platform.kill_process("NonExistentProcessXYZ_12345")
        assert isinstance(result, bool)

    def test_kill_process_by_invalid_pid(self, platform):
        """用无效 PID 杀进程返回 False"""
        result = platform.kill_process_by_pid(99999999)
        assert result is False


# ============================================================
# 兼容性检查
# ============================================================


class TestCompatibilityCheck:
    """兼容性检查测试"""

    def test_check_compatibility_returns_dict(self, platform):
        """返回包含所有字段的字典"""
        result = platform.check_compatibility()
        assert isinstance(result, dict)
        assert "compatible" in result
        assert "issues" in result
        assert "platform" in result
        assert "python_version" in result

    def test_compatible_is_bool(self, platform):
        """compatible 为布尔值"""
        result = platform.check_compatibility()
        assert isinstance(result["compatible"], bool)

    def test_issues_is_list(self, platform):
        """issues 为列表"""
        result = platform.check_compatibility()
        assert isinstance(result["issues"], list)

    def test_python_version_format(self, platform):
        """python_version 格式为 x.y.z"""
        result = platform.check_compatibility()
        parts = result["python_version"].split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()


# ============================================================
# WSL 检测
# ============================================================


class TestWSLDetection:
    """WSL 检测测试"""

    def test_is_wsl_returns_bool(self, platform):
        """is_wsl 返回布尔值"""
        assert isinstance(platform.is_wsl(), bool)

    def test_is_wsl_false_on_mac(self, platform):
        """macOS 上 is_wsl 为 False"""
        if platform.is_mac:
            assert platform.is_wsl() is False


# ============================================================
# 便捷函数
# ============================================================


class TestConvenienceFunctions:
    """便捷函数测试"""

    def test_kill_process_function(self):
        """kill_process 便捷函数"""
        from src.config.platform_compat import kill_process

        result = kill_process("NonExistentProcess")
        assert isinstance(result, bool)

    def test_ensure_dir_function(self, temp_dir):
        """ensure_dir 便捷函数"""
        from src.config.platform_compat import ensure_dir

        test_path = str(temp_dir / "conv_test")
        result = ensure_dir(test_path)
        assert result is True
        assert os.path.isdir(test_path)

    def test_run_command_function(self):
        """run_command 便捷函数"""
        from src.config.platform_compat import run_command

        result = run_command("echo convenience", shell=True)
        assert result["success"]
