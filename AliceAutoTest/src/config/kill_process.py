"""
进程清理模块 - 跨平台实现
使用psutil统一处理，最小化平台差异
"""

import ctypes
import os
import sys

from src.config.platform_compat import get_platform, is_windows, kill_process


def is_admin() -> bool:
    """检查是否有管理员权限"""
    return get_platform().is_admin()


def kill_phoenix_processes():
    """清理Phoenix相关进程"""
    phoenix_processes = [
        "Phoenix.exe",
        "phoenix.exe",
        "VCamDemo.exe",
        "VCamDemo",
        "AIBrowser.exe",
        "AIBrowser",
    ]

    print("开始清理Phoenix相关进程...")
    for proc_name in phoenix_processes:
        kill_process(proc_name)
    print("进程清理完成")


def clear_service():
    """清理服务（仅Windows）"""
    if not is_windows():
        print("非Windows系统，跳过服务清理")
        return

    print("清理服务...")
    try:
        result = get_platform().run_command(["net", "stop", "PhoenixService"])
        if result["success"]:
            print("服务清理完成")
        else:
            print(f"清理服务失败: {result['stderr']}")
    except Exception as e:
        print(f"清理服务失败: {e}")


def kill_by_pid_file(pid_file: str):
    """根据PID文件终止进程"""
    if not os.path.exists(pid_file):
        return

    try:
        with open(pid_file) as f:
            pid = int(f.read().strip())

        from src.config.config_hardware import ProcessManager

        pm = ProcessManager()
        if pm.kill_by_pid(pid):
            os.remove(pid_file)
            print(f"已终止PID {pid} 的进程")
        else:
            print(f"终止PID {pid} 失败")
    except Exception as e:
        print(f"读取PID文件失败: {e}")


def main():
    """主函数"""
    platform = get_platform()
    print(f"当前操作系统: {platform.system}")

    if is_admin():
        print("有管理员权限，执行完整清理...")
        kill_phoenix_processes()
        clear_service()
    else:
        if is_windows() and sys.version_info[0] == 3:
            # Windows: 请求管理员权限
            try:
                ctypes.windll.shell32.ShellExecuteW(
                    None, "runas", sys.executable, __file__, None, 1
                )
                return
            except Exception as e:
                print(f"请求管理员权限失败: {e}")

        # 无权限时的降级方案
        print("无管理员权限，仅清理当前用户进程...")
        kill_phoenix_processes()

    # 清理PID文件
    try:
        from src import settings

        pid_file = getattr(settings, "PID_PATH", "")
        if pid_file:
            kill_by_pid_file(pid_file)
    except Exception:
        pass

    print("清理完成")


if __name__ == "__main__":
    main()
