#!/usr/bin/env python
"""
Phoenix自动化测试框架 - 跨平台入口脚本
使用统一接口，最小化平台差异
"""

import os
import sys

# 添加项目根目录到路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


def print_banner():
    """打印欢迎信息"""
    print("=" * 60)
    print("  Phoenix自动化测试框架")
    print("  Phoenix Auto Test Framework")
    print("=" * 60)
    print()


def print_system_info():
    """打印系统信息"""
    from src.config.platform_compat import get_platform

    platform = get_platform()

    print("系统信息:")
    print(f"  操作系统: {platform.system}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  主机名: {platform.get_hostname()}")
    print(f"  IP地址: {platform.get_ip_address()}")
    print(f"  用户: {platform.get_user()}")
    print(f"  管理员权限: {'是' if platform.is_admin() else '否'}")
    print()


def main():
    """主函数"""
    print_banner()
    print_system_info()

    # 导入核心模块
    from src.config.base_file import BaseFile
    from src.config.config_hardware import PidConfig
    from src.config.config_logging import ConfigLogging
    from src.config.config_screen import ConfigScreen
    from src.config.platform_compat import is_windows
    from src.config.runtime import (
        cleanup_phoenix_processes,
        start_phoenix_client,
    )
    from src.modules.tools.three_hours import main as three_hours_main

    # 初始化
    from src.settings import ensure_directories

    ensure_directories()

    basefile = BaseFile()
    basefile.update_pid("1")

    # 备份日志
    logger = ConfigLogging()
    logger.handle_log()

    # 清除旧的截图文件
    screen = ConfigScreen()
    screen.del_screen_image()

    # 校验mock答题器文件
    kid = PidConfig()
    kid.copy_mockfile()

    # 清理进程（使用跨平台方法）
    cleanup_phoenix_processes(kid)

    import time

    time.sleep(2)

    # 启动学生端（仅Windows支持）
    if not start_phoenix_client(_PROJECT_ROOT):
        if is_windows():
            print("⚠️  未找到 config/start_phoenix.bat，跳过启动")
        else:
            print("⚠️  Phoenix学生端仅支持Windows系统，跳过启动")

    # 消息队列
    import multiprocessing
    from multiprocessing import Pipe, Process

    pipe = Pipe()

    # Windows下多进程必须加此代码
    if is_windows():
        multiprocessing.freeze_support()

    # 启动虚拟摄像头（仅Windows支持）
    pm_proc = None
    if is_windows():
        try:
            from tools.VCamTestTool.control2 import run_vcamtest

            kid.run_kill()
            pm_proc = Process(target=run_vcamtest, args=("e",))
            pm_proc.start()
            time.sleep(1)
        except Exception as e:
            print(f"⚠️  虚拟摄像头启动失败: {e}")
    else:
        print("⚠️  虚拟摄像头功能仅支持Windows系统，跳过")

    # 截图
    psc = Process(target=screen.run_screen)
    psc.start()

    # 三小时超时策略
    three = Process(target=three_hours_main)
    three.start()

    # 实例化主进程
    pw = None
    exit_code = 0
    try:
        from src.modules.px_run import PxPtAuto

        pea = PxPtAuto(pipe)

        # iPad端websocket
        pw = Process(target=pea.get_websocket_linking)
        pw.start()

        # 课中组件交互主程序
        pea.handle_info()

    except Exception as e:
        print(f"运行出错: {e}")
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        for process in (pw, pm_proc, psc, three):
            if process is not None and process.is_alive():
                process.terminate()
                process.join(timeout=5)
        cleanup_phoenix_processes(kid)

    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
