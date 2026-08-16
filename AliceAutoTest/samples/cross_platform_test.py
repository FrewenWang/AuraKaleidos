#!/usr/bin/env python
"""
Phoenix自动化测试框架 - 跨平台功能验证 Demo

本 Demo 全面验证框架的跨平台兼容性，自动检测当前操作系统
并逐一测试所有跨平台功能模块：

1. 平台检测与系统信息
2. 路径管理（用户目录/应用数据/临时目录）
3. 目录创建与清理
4. 进程管理（进程列表/杀进程）
5. 文件操作（创建/复制/删除）
6. 硬件信息获取（主机名/IP/MAC）
7. 跨平台兼容性检查
8. 配置文件读取
9. 日志记录
10. JSON/TXT 文件读写
11. 截图功能（需 GUI 环境）
12. 管理员权限检测

运行方式:
    python samples/cross_platform_test.py
"""

import importlib.util
import os
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


# ============================================================
# 测试结果统计
# ============================================================

_test_results = []


def record_result(name, passed, detail=""):
    """记录测试结果"""
    status = "✅ 通过" if passed else "❌ 失败"
    _test_results.append((name, passed, detail))
    print(f"  {status} | {name}")
    if detail and not passed:
        print(f"         详情: {detail}")


def print_section(title):
    """打印测试区块标题"""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ============================================================
# 测试用例
# ============================================================


def test_platform_detection():
    """测试1: 平台检测"""
    print_section("测试1: 平台检测")
    try:
        from src.config.platform_compat import (
            get_platform,
            is_linux,
            is_mac,
            is_windows,
        )

        platform = get_platform()

        # 验证平台属性
        assert platform.system in ("windows", "darwin", "linux"), (
            f"未知的操作系统: {platform.system}"
        )

        # 验证平台判断函数一致性
        if platform.is_windows:
            assert is_windows() is True
            assert is_mac() is False
            assert is_linux() is False
        elif platform.is_mac:
            assert is_windows() is False
            assert is_mac() is True
            assert is_linux() is False
        elif platform.is_linux:
            assert is_windows() is False
            assert is_mac() is False
            assert is_linux() is True

        print(f"  操作系统: {platform.system}")
        print(
            f"  is_windows: {is_windows()}  is_mac: {is_mac()}  is_linux: {is_linux()}"
        )

        record_result("平台检测", True)
    except Exception as e:
        record_result("平台检测", False, str(e))


def test_system_info():
    """测试2: 系统信息获取"""
    print_section("测试2: 系统信息获取")
    try:
        from src.config.platform_compat import get_platform

        platform = get_platform()

        # 用户名
        user = platform.get_user()
        assert user and len(user) > 0, "用户名为空"
        print(f"  当前用户: {user}")

        # 主机名
        hostname = platform.get_hostname()
        assert hostname and len(hostname) > 0, "主机名为空"
        print(f"  主机名: {hostname}")

        # Python 版本
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        print(f"  Python: {py_ver}")

        # 管理员权限
        admin = platform.is_admin()
        print(f"  管理员权限: {'是' if admin else '否'}")

        record_result("系统信息获取", True)
    except Exception as e:
        record_result("系统信息获取", False, str(e))


def test_path_management():
    """测试3: 路径管理"""
    print_section("测试3: 路径管理")
    try:
        from src.config.platform_compat import get_platform, is_windows

        platform = get_platform()

        # 用户主目录
        user_home = platform.get_user_home()
        assert os.path.isdir(user_home), f"用户主目录不存在: {user_home}"
        print(f"  用户主目录: {user_home}")

        # 应用数据目录
        app_data = platform.get_app_data_path("Phoenix")
        print(f"  应用数据目录: {app_data}")

        # 验证路径格式符合当前平台
        if is_windows():
            assert ":" in app_data or "\\\\" in app_data, (
                f"Windows路径格式异常: {app_data}"
            )
        else:
            assert app_data.startswith("/"), f"Unix路径格式异常: {app_data}"

        # 临时目录
        temp_path = platform.get_temp_path()
        assert os.path.isdir(temp_path), f"临时目录不存在: {temp_path}"
        print(f"  临时目录: {temp_path}")

        # settings.py 路径常量
        from src.settings import LOG_PATH, P_P, PHOENIX_BIN, PID_NAME, PID_PATH

        print(f"  Phoenix基础路径 (P_P): {P_P}")
        print(f"  Phoenix bin目录: {PHOENIX_BIN}")
        print(f"  日志路径: {LOG_PATH}")
        print(f"  PID文件: {PID_PATH}")
        print(f"  进程名: {PID_NAME}")

        record_result("路径管理", True)
    except Exception as e:
        record_result("路径管理", False, str(e))


def test_directory_creation():
    """测试4: 目录创建"""
    print_section("测试4: 目录创建")
    try:
        from src.config.platform_compat import get_platform

        platform = get_platform()
        test_base = os.path.join(
            platform.get_temp_path(), "phoenix_cross_platform_test"
        )

        # 创建多级目录
        test_dir = os.path.join(test_base, "level1", "level2", "level3")
        assert platform.ensure_dir(test_dir), "目录创建失败"
        assert os.path.isdir(test_dir), f"目录不存在: {test_dir}"
        print(f"  创建多级目录: {test_dir}")

        # ensure_dir 幂等性（重复创建不应报错）
        assert platform.ensure_dir(test_dir), "重复创建目录失败"
        print("  重复创建目录: OK（幂等）")

        # 清理
        import shutil

        shutil.rmtree(test_base, ignore_errors=True)
        print("  清理测试目录: OK")

        record_result("目录创建", True)
    except Exception as e:
        record_result("目录创建", False, str(e))


def test_file_operations():
    """测试5: 文件操作"""
    print_section("测试5: 文件操作")
    try:
        from src.config.platform_compat import get_platform

        platform = get_platform()
        test_dir = os.path.join(platform.get_temp_path(), "phoenix_file_test")
        platform.ensure_dir(test_dir)

        # 创建文件
        src_file = os.path.join(test_dir, "source.txt")
        with open(src_file, "w", encoding="utf-8") as f:
            f.write("Hello Phoenix Cross-Platform Test\n跨平台文件测试\n")
        assert os.path.isfile(src_file), "源文件创建失败"
        print(f"  创建文件: {src_file}")

        # 复制文件
        dst_file = os.path.join(test_dir, "copied.txt")
        assert platform.copy_file(src_file, dst_file), "文件复制失败"
        assert os.path.isfile(dst_file), "目标文件不存在"
        print(f"  复制文件: {dst_file}")

        # 验证内容一致
        with open(src_file, encoding="utf-8") as f:
            src_content = f.read()
        with open(dst_file, encoding="utf-8") as f:
            dst_content = f.read()
        assert src_content == dst_content, "文件内容不一致"

        # 清理
        import shutil

        shutil.rmtree(test_dir, ignore_errors=True)
        print("  清理测试文件: OK")

        record_result("文件操作", True)
    except Exception as e:
        record_result("文件操作", False, str(e))


def test_hardware_info():
    """测试6: 硬件信息获取"""
    print_section("测试6: 硬件信息获取")
    try:
        from src.config.config_hardware import HardwareConfig

        hw = HardwareConfig()

        # 主机名
        hostname = hw.get_hostname()
        assert hostname, "主机名为空"
        print(f"  主机名: {hostname}")

        # IP 地址
        ip = hw.platform.get_ip_address()
        assert ip, "IP地址为空"
        print(f"  IP地址: {ip}")

        # MAC 地址
        mac = hw.get_mac()
        assert mac, "MAC地址为空"
        print(f"  MAC地址: {mac}")

        # 进程数量
        import psutil

        proc_count = len(psutil.pids())
        print(f"  进程数量: {proc_count}")

        record_result("硬件信息获取", True)
    except Exception as e:
        record_result("硬件信息获取", False, str(e))


def test_compatibility_check():
    """测试7: 兼容性检查"""
    print_section("测试7: 兼容性检查")
    try:
        from src.config.platform_compat import get_platform

        platform = get_platform()
        result = platform.check_compatibility()

        assert "compatible" in result, "返回结果缺少 compatible 字段"
        assert "issues" in result, "返回结果缺少 issues 字段"
        assert "platform" in result, "返回结果缺少 platform 字段"
        assert "python_version" in result, "返回结果缺少 python_version 字段"

        print(f"  平台: {result['platform']}")
        print(f"  Python版本: {result['python_version']}")
        print(f"  兼容: {result['compatible']}")

        if result["issues"]:
            for issue in result["issues"]:
                print(f"  ⚠️  {issue}")

        record_result("兼容性检查", True)
    except Exception as e:
        record_result("兼容性检查", False, str(e))


def test_config_reading():
    """测试8: 配置文件读取"""
    print_section("测试8: 配置文件读取")
    try:
        from src.config.base_config import ReadBaseConfig

        config = ReadBaseConfig()

        # HTTP 配置
        login_url = config.get_http("base_login_url")
        assert login_url, "base_login_url 为空"
        print(f"  登录URL: {login_url}")

        # LOGIN 配置
        username = config.get_login("en_username")
        assert username, "en_username 为空"
        print(f"  英语用户名: {username}")

        # DB 配置
        db_host = config.get_db("host")
        assert db_host, "DB host 为空"
        print(f"  数据库地址: {db_host}")

        record_result("配置文件读取", True)
    except Exception as e:
        record_result("配置文件读取", False, str(e))


def test_logging():
    """测试9: 日志记录"""
    print_section("测试9: 日志记录")
    try:
        from src.config.config_logging import ConfigLogging

        logger = ConfigLogging().write_logging()

        logger.debug("[Cross-Platform Test] DEBUG 级别日志")
        logger.info("[Cross-Platform Test] INFO 级别日志")
        logger.warning("[Cross-Platform Test] WARNING 级别日志")
        logger.error("[Cross-Platform Test] ERROR 级别日志")

        print("  日志级别: DEBUG / INFO / WARNING / ERROR 全部写入")

        record_result("日志记录", True)
    except Exception as e:
        record_result("日志记录", False, str(e))


def test_json_txt_fileconfig():
    """测试10: JSON/TXT 文件读写"""
    print_section("测试10: JSON/TXT 文件读写")
    try:
        from src.config.platform_compat import get_platform

        platform = get_platform()
        test_dir = os.path.join(
            platform.get_temp_path(), "phoenix_fileconfig_test"
        )
        platform.ensure_dir(test_dir)

        # === JSON 测试 ===
        # HandleJson 继承 JsonFileHandler，通用读写方法为 load_json / save_json
        # read_json 是读取 Phoenix 组件文件的专用方法，不用于通用文件读取
        from src.fileconfig.file_config_base import JsonFileHandler

        json_handler = JsonFileHandler()
        json_file = os.path.join(test_dir, "test.json")
        json_data = {
            "platform": platform.system,
            "hostname": platform.get_hostname(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "items": ["item1", "item2", "item3"],
            "nested": {"key": "value", "number": 42},
        }
        json_handler.save_json(json_file, json_data)
        print(f"  JSON写入: {json_file}")

        read_data = json_handler.load_json(json_file)
        assert read_data["platform"] == json_data["platform"], (
            "JSON platform 不匹配"
        )
        assert read_data["items"] == json_data["items"], "JSON items 不匹配"
        assert read_data["nested"]["number"] == 42, "JSON nested.number 不匹配"
        print("  JSON读取: 验证通过")

        # === TXT 测试 ===
        # TxtFileConfig 继承 TxtFileHandler → BaseFileHandler
        # write_file(filepath, data, mode)  read_file(filepath) 返回按行分割的列表
        from src.fileconfig.file_config_base import TxtFileHandler

        txt_handler = TxtFileHandler()
        txt_file = os.path.join(test_dir, "test.txt")
        txt_content = "第一行: 跨平台测试\n第二行: 文件读写\n第三行: 结束\n"
        txt_handler.write_file(txt_file, txt_content, mode="w")
        print(f"  TXT写入: {txt_file}")

        read_lines = txt_handler.read_file(txt_file)
        # read_file 返回按 \n 分割的列表
        all_text = "\n".join(read_lines) if read_lines else ""
        assert "跨平台测试" in all_text, "TXT 内容不匹配"
        assert "文件读写" in all_text, "TXT 内容不匹配"
        print(f"  TXT读取: 验证通过 ({len(read_lines)} 行)")

        # 清理
        import shutil

        shutil.rmtree(test_dir, ignore_errors=True)
        print("  清理: OK")

        record_result("JSON/TXT 文件读写", True)
    except Exception as e:
        record_result("JSON/TXT 文件读写", False, str(e))


def test_screen_capture():
    """测试11: 截图功能"""
    print_section("测试11: 截图功能")
    try:
        # 先检查可选依赖。config_screen 在模块加载时会导入
        # NumPy/OpenCV，所以必须在导入 ConfigScreen 前完成降级判断。
        if importlib.util.find_spec("PyQt5") is not None:
            pyqt5_available = True
            print("  PyQt5: 已安装 (可截图)")
        else:
            pyqt5_available = False
            print("  PyQt5: 未安装 (截图功能不可用)")

        # 检查 OpenCV 是否可用
        try:
            import cv2

            cv2_available = True
            print(f"  OpenCV: 已安装 (可图片匹配) v{cv2.__version__}")
        except ImportError:
            cv2_available = False
            print("  OpenCV: 未安装 (图片匹配不可用)")

        if not pyqt5_available or not cv2_available:
            missing = []
            if not pyqt5_available:
                missing.append("PyQt5")
            if not cv2_available:
                missing.append("OpenCV/NumPy")
            print(f"  跳过截图实测：缺少可选依赖 {', '.join(missing)}")
            record_result("截图功能（可选依赖未安装，已跳过）", True)
            return

        from src.config.config_screen import ConfigScreen

        screen = ConfigScreen()
        screen.del_screen_image()
        print("  旧截图清理: OK")
        record_result("截图功能", True)
    except Exception as e:
        record_result("截图功能", False, str(e))


def test_process_management():
    """测试12: 进程管理"""
    print_section("测试12: 进程管理")
    try:
        import psutil

        from src.config.platform_compat import get_platform

        platform = get_platform()

        # 获取进程列表
        pids = psutil.pids()
        assert len(pids) > 0, "进程列表为空"
        print(f"  当前进程数: {len(pids)}")

        # 获取当前进程信息
        current_proc = psutil.Process()
        print(f"  当前进程 PID: {current_proc.pid}")
        print(f"  当前进程名: {current_proc.name()}")

        # 测试 kill_process（用一个不存在的进程名，不应报错）
        result = platform.kill_process("NonExistentProcess12345")
        print(f"  杀不存在的进程: 返回 {result}（不应报错）")

        record_result("进程管理", True)
    except Exception as e:
        record_result("进程管理", False, str(e))


def test_command_execution():
    """测试13: 命令执行"""
    print_section("测试13: 命令执行")
    try:
        from src.config.platform_compat import get_platform, is_windows

        platform = get_platform()

        # 执行跨平台命令
        cmd = "echo hello" if is_windows() else "echo hello"

        result = platform.run_command(cmd, shell=True)
        assert result["success"], f"命令执行失败: {result['stderr']}"
        assert "hello" in result["stdout"], (
            f"输出不包含hello: {result['stdout']}"
        )
        print(f"  命令: {cmd}")
        print(f"  输出: {result['stdout'].strip()}")
        print(f"  返回码: {result['returncode']}")

        record_result("命令执行", True)
    except Exception as e:
        record_result("命令执行", False, str(e))


def test_settings_constants():
    """测试14: 全局设置常量"""
    print_section("测试14: 全局设置常量")
    try:
        from src.settings import (
            LOG_PATH,
            P_P,
            PHOENIX_BIN,
            PID_NAME,
            PID_PATH,
            TIME,
            ensure_directories,
            get_status_info,
        )

        # 验证常量类型
        assert isinstance(P_P, (str, Path)), "P_P 类型异常"
        assert isinstance(PID_NAME, str), "PID_NAME 类型异常"
        assert isinstance(TIME, int), "TIME 类型异常"

        print(f"  P_P: {P_P}")
        print(f"  PHOENIX_BIN: {PHOENIX_BIN}")
        print(f"  LOG_PATH: {LOG_PATH}")
        print(f"  PID_PATH: {PID_PATH}")
        print(f"  PID_NAME: {PID_NAME}")
        print(f"  超时时间: {TIME}秒 ({TIME // 3600}小时)")

        # 验证 ensure_directories 不报错
        ensure_directories()
        print("  ensure_directories: OK")

        # 验证 get_status_info
        info = get_status_info()
        assert "os" in info, "状态信息缺少 os"
        assert "is_windows" in info, "状态信息缺少 is_windows"
        print(f"  get_status_info: {len(info)} 个字段")

        record_result("全局设置常量", True)
    except Exception as e:
        record_result("全局设置常量", False, str(e))


# ============================================================
# 主函数
# ============================================================


def main():
    """运行所有跨平台测试"""
    print("\n" + "=" * 60)
    print("  Phoenix 自动化测试框架 - 跨平台功能验证")
    print("  Cross-Platform Compatibility Test")
    print("=" * 60)

    from src.config.platform_compat import get_platform

    platform = get_platform()
    print(f"\n  当前平台: {platform.system}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  主机: {platform.get_hostname()}")
    print(f"  用户: {platform.get_user()}")

    # 运行所有测试
    tests = [
        test_platform_detection,
        test_system_info,
        test_path_management,
        test_directory_creation,
        test_file_operations,
        test_hardware_info,
        test_compatibility_check,
        test_config_reading,
        test_logging,
        test_json_txt_fileconfig,
        test_screen_capture,
        test_process_management,
        test_command_execution,
        test_settings_constants,
    ]

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            record_result(
                test_func.__doc__ or test_func.__name__, False, str(e)
            )

    # 汇总结果
    print("\n" + "=" * 60)
    print("  测试结果汇总")
    print("=" * 60)

    passed = sum(1 for _, p, _ in _test_results if p)
    failed = sum(1 for _, p, _ in _test_results if not p)
    total = len(_test_results)

    for name, p, detail in _test_results:
        status = "✅" if p else "❌"
        print(f"  {status} {name}")
        if not p and detail:
            print(f"      → {detail}")

    print(f"\n  总计: {total}  通过: {passed}  失败: {failed}")
    print(f"  通过率: {passed / total * 100:.1f}%")

    if failed == 0:
        print("\n  🎉 所有跨平台测试通过！")
    else:
        print(f"\n  ⚠️  有 {failed} 项测试未通过，请检查详情。")

    print("=" * 60 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
