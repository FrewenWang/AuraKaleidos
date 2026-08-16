"""
文件配置模块 - 统一的文件操作接口
"""

from .file_config_base import (
    BaseFileHandler,
    JsonFileHandler,
    TxtFileHandler,
    ensure_dir,
    read_file,
    write_file,
)

__all__ = [
    "BaseFileHandler",
    "JsonFileHandler",
    "TxtFileHandler",
    "read_file",
    "write_file",
    "ensure_dir",
]
