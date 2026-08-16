"""
文件配置基类 - 统一的文件操作接口
消除重复代码，提供一致的文件操作API
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any

from src.config.base_config import ReadBaseConfig
from src.config.datetime_utils import ConfigTime


class BaseFileHandler:
    """文件操作基类"""

    def __init__(self):
        self.config = ReadBaseConfig()
        self.datetime = ConfigTime().reporttime

    def get_report_path(self, name: str = "reportpath") -> Path:
        """获取报告路径"""
        return Path(self.config.get_reportpath(name))

    def get_filepath(self, name: str) -> Path:
        """获取文件路径"""
        return Path(self.config.get_filepath(name))

    def read_file(
        self, filepath: str, encoding: str = "utf-8"
    ) -> list[str] | None:
        """读取文件内容"""
        try:
            with open(filepath, encoding=encoding) as f:
                content = f.read()
            return content.split("\n")
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"读取文件失败 {filepath}: {e}")
            return None

    def write_file(
        self, filepath: str, data: str, mode: str = "a", encoding: str = "utf-8"
    ) -> bool:
        """写入文件"""
        try:
            with open(filepath, mode, encoding=encoding) as f:
                f.write(data)
            return True
        except Exception as e:
            print(f"写入文件失败 {filepath}: {e}")
            return False

    def append_line(
        self, filepath: str, data: str, encoding: str = "utf-8"
    ) -> bool:
        """追加一行数据"""
        return self.write_file(
            filepath, f"{self.datetime}: {data}\n", "a", encoding
        )

    def list_files(self, directory: str) -> list[str]:
        """列出目录下的文件"""
        try:
            return os.listdir(directory)
        except FileNotFoundError:
            return []

    def get_latest_file(self, directory: str) -> Path | None:
        """获取目录下最新的文件"""
        try:
            files = list(Path(directory).iterdir())
            if not files:
                return None
            latest = max(files, key=lambda f: f.stat().st_mtime)
            return latest
        except Exception:
            return None

    def copy_file(self, source: str, destination: str) -> bool:
        """复制文件"""
        try:
            shutil.copy2(source, destination)
            return True
        except Exception as e:
            print(f"复制文件失败: {e}")
            return False

    def move_file(self, source: str, destination: str) -> bool:
        """移动文件"""
        try:
            shutil.move(source, destination)
            return True
        except Exception as e:
            print(f"移动文件失败: {e}")
            return False

    def delete_file(self, filepath: str) -> bool:
        """删除文件"""
        try:
            os.remove(filepath)
            return True
        except Exception as e:
            print(f"删除文件失败: {e}")
            return False

    def ensure_dir(self, path: str) -> bool:
        """确保目录存在"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            print(f"创建目录失败: {e}")
            return False

    def get_file_extension(self, filename: str) -> str:
        """获取文件扩展名"""
        return Path(filename).suffix

    def get_filename_without_ext(self, filename: str) -> str:
        """获取不带扩展名的文件名"""
        return Path(filename).stem


class JsonFileHandler(BaseFileHandler):
    """JSON文件处理器"""

    def __init__(self):
        super().__init__()
        self._json_data = None

    def load_json(self, filepath: str) -> Any:
        """加载JSON文件"""
        try:
            with open(filepath, encoding="utf-8") as f:
                self._json_data = json.load(f)
            return self._json_data
        except Exception as e:
            print(f"加载JSON失败: {e}")
            return None

    def save_json(self, filepath: str, data: Any) -> bool:
        """保存JSON文件"""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存JSON失败: {e}")
            return False


class TxtFileHandler(BaseFileHandler):
    """文本文件处理器"""

    def __init__(self):
        super().__init__()
        self.report_path = self.get_report_path("reportpath")

    def write_log(self, data: str) -> bool:
        """写入日志"""
        log_path = self.report_path / "log.txt"
        return self.append_line(str(log_path), data)

    def read_log(self) -> list[str] | None:
        """读取日志"""
        log_path = self.report_path / "log.txt"
        return self.read_file(str(log_path))


# 便捷函数
def read_file(filepath: str, encoding: str = "utf-8") -> list[str] | None:
    """便捷函数：读取文件"""
    handler = BaseFileHandler()
    return handler.read_file(filepath, encoding)


def write_file(
    filepath: str, data: str, mode: str = "a", encoding: str = "utf-8"
) -> bool:
    """便捷函数：写入文件"""
    handler = BaseFileHandler()
    return handler.write_file(filepath, data, mode, encoding)


def ensure_dir(path: str) -> bool:
    """便捷函数：确保目录存在"""
    handler = BaseFileHandler()
    return handler.ensure_dir(path)
