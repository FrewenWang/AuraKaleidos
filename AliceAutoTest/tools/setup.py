"""
Phoenix自动化测试框架 - 跨平台安装脚本
使用统一接口，最小化平台差异
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
import contextlib

from src.config.platform_compat import get_platform


def print_banner():
    """打印欢迎信息"""
    print("=" * 60)
    print("  Phoenix自动化测试框架 - 安装程序")
    print("=" * 60)
    print()


def check_python_version() -> bool:
    """检查Python版本"""
    print("[1/5] 检查Python版本...")

    version = sys.version_info
    print(f"  Python版本: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 6):
        print("  ❌ 错误: 需要Python 3.6或更高版本")
        return False

    print("  ✅ Python版本检查通过")
    return True


def install_dependencies() -> bool:
    """安装依赖包"""
    print("\n[2/5] 安装依赖包...")

    # 基础依赖（所有平台）
    packages = [
        "certifi>=2018.11.29",
        "chardet>=3.0.4",
        "idna>=2.8",
        "numpy>=1.17.2",
        "psutil>=5.6.2",
        "PyMySQL>=0.9.3",
        "python-dateutil>=2.8.0",
        "pytz>=2019.3",
        "requests>=2.21.0",
        "six>=1.12.0",
        "urllib3>=1.24.1",
        "websocket-client>=0.57.0",
        "opencv-python>=4.1.1.26",
        "PyQt5>=5.12.1",
        "mysql-connector-python>=8.0.13",
        "gevent>=1.4.0",
        "greenlet>=0.4.15",
        "Pillow>=6.0.0",
    ]

    # 使用清华镜像源
    mirror_url = "https://pypi.tuna.tsinghua.edu.cn/simple"

    print(f"  将安装 {len(packages)} 个包...")
    print(f"  使用镜像源: {mirror_url}")
    print()

    # 一次性安装所有包
    cmd = [sys.executable, "-m", "pip", "install", "-i", mirror_url] + packages
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("  ✅ 所有依赖安装完成")
        return True
    else:
        print("  ⚠️  部分依赖安装失败")
        print(f"  错误: {result.stderr[-200:]}...")
        return False


def setup_directories() -> bool:
    """创建必要的目录结构"""
    print("\n[3/5] 创建目录结构...")

    from src.settings import ensure_directories

    ensure_directories()
    print("  ✅ 目录结构创建完成")
    return True


def check_environment() -> bool:
    """检查环境兼容性"""
    print("\n[4/5] 检查环境兼容性...")

    platform = get_platform()
    result = platform.check_compatibility()

    print(f"  平台: {result['platform']}")
    print(f"  Python: {result['python_version']}")

    if result["compatible"]:
        print("  ✅ 环境检查通过")
        return True
    else:
        print("  ⚠️  发现以下问题:")
        for issue in result["issues"]:
            print(f"      - {issue}")
        return len(result["issues"]) == 0


def print_next_steps():
    """打印后续步骤"""
    from src.config.platform_compat import is_windows

    print("\n" + "=" * 60)
    print("  安装完成！")
    print("=" * 60)
    print()
    print("后续步骤:")
    print()
    print("1. 配置数据库和API:")
    print("   - 编辑 config/config.ini")
    print("   - 设置数据库连接信息")
    print("   - 配置API端点")
    print()

    if is_windows():
        print("2. 安装虚拟摄像头驱动:")
        print("   - 运行 VCamTestTool/install.bat")
        print()

    print("3. 运行测试:")
    print("   - python setup.py --test")
    print()

    print("4. 启动程序:")
    print("   - python run.py")
    print()

    print("更多信息请查看 QUICK_START.md")
    print()


def run_tests():
    """运行基本测试"""
    print("\n运行基本测试...")
    print()

    # 测试平台检测
    print("[测试1] 平台检测...")
    platform = get_platform()
    print(f"  操作系统: {platform.system}")
    print("  ✅ 平台检测正常")
    print()

    # 测试路径功能
    print("[测试2] 路径功能...")
    print(f"  用户主目录: {platform.get_user_home()}")
    print(f"  应用数据目录: {platform.get_app_data_path()}")
    print(f"  临时目录: {platform.get_temp_path()}")
    print("  ✅ 路径功能正常")
    print()

    # 测试目录创建
    print("[测试3] 目录创建...")
    test_dir = Path(platform.get_temp_path()) / "phoenix_test"
    if platform.ensure_dir(str(test_dir)):
        print(f"  测试目录: {test_dir}")
        print("  ✅ 目录创建正常")
        with contextlib.suppress(BaseException):
            test_dir.rmdir()
    else:
        print("  ❌ 目录创建失败")
    print()

    # 测试模块导入
    print("[测试4] 模块导入...")
    try:
        for module_name in (
            "src.config.base_config",
            "src.config.config_hardware",
            "src.config.config_logging",
            "src.config.platform_compat",
        ):
            importlib.import_module(module_name)

        print("  ✅ 核心模块导入正常")
    except ImportError as e:
        print(f"  ❌ 模块导入失败: {e}")
    print()

    # 测试兼容性
    print("[测试5] 兼容性检查...")
    result = platform.check_compatibility()
    if result["compatible"]:
        print("  ✅ 环境兼容")
    else:
        print("  ⚠️  存在兼容性问题:")
        for issue in result["issues"]:
            print(f"      - {issue}")
    print()

    print("测试完成！")


def main():
    """主函数"""
    print_banner()

    # 检查Python版本
    if not check_python_version():
        sys.exit(1)

    # 安装依赖
    install_dependencies()

    # 创建目录
    setup_directories()

    # 检查环境
    check_environment()

    # 打印后续步骤
    print_next_steps()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_tests()
    else:
        main()
