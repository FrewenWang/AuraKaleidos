"""
pytest 公共夹具
为所有测试提供项目根目录路径注入和通用 fixtures
"""

import sys
from pathlib import Path

# 将项目根目录添加到 sys.path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pytest


@pytest.fixture
def project_root():
    """项目根目录"""
    return _PROJECT_ROOT


@pytest.fixture
def temp_dir(tmp_path):
    """临时测试目录（pytest 自动清理）"""
    return tmp_path


@pytest.fixture
def platform():
    """跨平台兼容实例"""
    from src.config.platform_compat import get_platform

    return get_platform()


@pytest.fixture
def config():
    """配置读取器实例"""
    from src.config.base_config import ReadBaseConfig

    return ReadBaseConfig()
