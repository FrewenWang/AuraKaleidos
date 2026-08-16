#!/usr/bin/env python
"""
Phoenix自动化测试框架 - 基础使用示例

本示例展示如何使用框架的基础功能：
1. 配置读取
2. 日志记录
3. 文件操作
4. 数据库操作
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_config_usage():
    """示例1: 配置读取"""
    print("=" * 60)
    print("示例1: 配置读取")
    print("=" * 60)

    from src.config.base_config import ReadBaseConfig

    # 创建配置读取器
    config = ReadBaseConfig()

    # 读取HTTP配置
    base_url = config.get_http("base_login_url")
    print(f"基础URL: {base_url}")

    # 读取登录配置
    username = config.get_login("en_username")
    print(f"英语用户名: {username}")

    print("✓ 配置读取成功\n")


def example_logging():
    """示例2: 日志记录"""
    print("=" * 60)
    print("示例2: 日志记录")
    print("=" * 60)

    from src.config.config_logging import ConfigLogging

    # 创建日志记录器
    logger = ConfigLogging().write_logging()

    # 记录不同级别的日志
    logger.debug("这是调试信息")
    logger.info("这是普通信息")
    logger.warning("这是警告信息")
    logger.error("这是错误信息")

    print("✓ 日志记录成功\n")


def example_file_operations():
    """示例3: 文件操作"""
    print("=" * 60)
    print("示例3: 文件操作")
    print("=" * 60)

    from src.fileconfig.json_file_config import HandleJson
    from src.fileconfig.txt_file_config import TxtFileConfig

    # JSON文件操作
    json_handler = HandleJson()
    json_data = {"name": "测试", "value": 123}
    json_handler.save_json("test.json", json_data)
    print("✓ JSON文件写入成功")

    # TXT文件操作
    txt_handler = TxtFileConfig()
    txt_handler.write_file("test.txt", "行1\n行2\n行3\n")
    print("✓ TXT文件写入成功")

    # 清理测试文件
    os.remove("test.json")
    os.remove("test.txt")
    print("✓ 测试文件已清理\n")


def example_hardware_info():
    """示例4: 硬件信息获取"""
    print("=" * 60)
    print("示例4: 硬件信息获取")
    print("=" * 60)

    from src.config.config_hardware import HardwareConfig

    hw_config = HardwareConfig()

    # 获取主机名
    hostname = hw_config.get_hostname()
    print(f"主机名: {hostname}")

    # 获取IP地址
    ip = hw_config.platform.get_ip_address()
    print(f"IP地址: {ip}")

    # 获取MAC地址
    mac = hw_config.get_mac()
    print(f"MAC地址: {mac}")

    print("✓ 硬件信息获取成功\n")


def example_platform_compat():
    """示例5: 跨平台兼容"""
    print("=" * 60)
    print("示例5: 跨平台兼容")
    print("=" * 60)

    from src.config.platform_compat import get_platform, is_macos, is_windows

    platform = get_platform()

    print(f"操作系统: {platform.system}")
    print(f"是否Windows: {is_windows()}")
    print(f"是否macOS: {is_macos()}")
    print(f"用户目录: {platform.get_user_home()}")

    # 获取应用数据路径（跨平台）
    app_data = platform.get_app_data_path()
    print(f"应用数据路径: {app_data}")

    print("✓ 跨平台兼容演示成功\n")


def example_screen_capture():
    """示例6: 截图功能"""
    print("=" * 60)
    print("示例6: 截图功能")
    print("=" * 60)

    from src.config.config_screen import ConfigScreen

    screen = ConfigScreen()

    # 删除旧截图
    screen.del_screen_image()
    print("✓ 旧截图已清理")

    print("提示: 完整截图功能需要在GUI环境中运行")
    print("✓ 截图功能演示成功\n")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("  Phoenix自动化测试框架 - 基础使用示例")
    print("=" * 60 + "\n")

    try:
        # 运行各个示例
        example_config_usage()
        example_logging()
        example_file_operations()
        example_hardware_info()
        example_platform_compat()
        example_screen_capture()

        print("=" * 60)
        print("  所有示例运行成功！")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ 运行出错: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
