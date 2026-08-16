"""
文件配置模块 (fileconfig) 单元测试

测试 JSON 和 TXT 文件读写功能
"""

import os

import pytest


class TestJsonFileHandler:
    """JSON 文件处理器测试"""

    @pytest.fixture
    def handler(self):
        from src.fileconfig.file_config_base import JsonFileHandler

        return JsonFileHandler()

    def test_save_and_load_json(self, handler, temp_dir):
        """保存并加载 JSON"""
        filepath = str(temp_dir / "test.json")
        data = {"name": "Phoenix", "version": 2, "items": [1, 2, 3]}

        assert handler.save_json(filepath, data)
        loaded = handler.load_json(filepath)
        assert loaded == data

    def test_save_json_with_unicode(self, handler, temp_dir):
        """保存含中文的 JSON"""
        filepath = str(temp_dir / "unicode.json")
        data = {"名称": "凤凰", "描述": "自动化测试"}

        assert handler.save_json(filepath, data)
        loaded = handler.load_json(filepath)
        assert loaded["名称"] == "凤凰"

    def test_save_json_with_nested(self, handler, temp_dir):
        """保存嵌套 JSON"""
        filepath = str(temp_dir / "nested.json")
        data = {"level1": {"level2": {"level3": "deep_value"}}}

        assert handler.save_json(filepath, data)
        loaded = handler.load_json(filepath)
        assert loaded["level1"]["level2"]["level3"] == "deep_value"

    def test_load_nonexistent_json(self, handler, temp_dir):
        """加载不存在的 JSON 文件返回 None"""
        filepath = str(temp_dir / "nonexistent.json")
        result = handler.load_json(filepath)
        assert result is None

    def test_save_json_creates_file(self, handler, temp_dir):
        """save_json 确实创建了文件"""
        filepath = str(temp_dir / "created.json")
        handler.save_json(filepath, {"key": "value"})
        assert os.path.isfile(filepath)


class TestTxtFileHandler:
    """TXT 文件处理器测试"""

    @pytest.fixture
    def handler(self):
        from src.fileconfig.file_config_base import TxtFileHandler

        return TxtFileHandler()

    def test_write_and_read_file(self, handler, temp_dir):
        """写入并读取文件"""
        filepath = str(temp_dir / "test.txt")
        content = "line one\nline two\nline three\n"

        assert handler.write_file(filepath, content, mode="w")
        lines = handler.read_file(filepath)
        assert lines is not None
        # read_file 返回按 \n 分割的列表
        assert len(lines) >= 3

    def test_write_file_append_mode(self, handler, temp_dir):
        """追加模式写入"""
        filepath = str(temp_dir / "append.txt")

        handler.write_file(filepath, "first\n", mode="w")
        handler.write_file(filepath, "second\n", mode="a")

        lines = handler.read_file(filepath)
        all_text = "\n".join(lines)
        assert "first" in all_text
        assert "second" in all_text

    def test_read_nonexistent_file(self, handler, temp_dir):
        """读取不存在的文件返回 None"""
        filepath = str(temp_dir / "nonexistent.txt")
        result = handler.read_file(filepath)
        assert result is None

    def test_write_unicode(self, handler, temp_dir):
        """写入中文内容"""
        filepath = str(temp_dir / "unicode.txt")
        content = "第一行\n第二行\n"

        assert handler.write_file(filepath, content, mode="w")
        lines = handler.read_file(filepath)
        all_text = "\n".join(lines)
        assert "第一行" in all_text
        assert "第二行" in all_text


class TestBaseFileHandler:
    """BaseFileHandler 测试"""

    @pytest.fixture
    def handler(self):
        from src.fileconfig.file_config_base import BaseFileHandler

        return BaseFileHandler()

    def test_copy_file(self, handler, temp_dir):
        """复制文件"""
        src = temp_dir / "src.txt"
        dst = temp_dir / "dst.txt"
        src.write_text("copy test")

        assert handler.copy_file(str(src), str(dst))
        assert dst.exists()
        assert dst.read_text() == "copy test"

    def test_move_file(self, handler, temp_dir):
        """移动文件"""
        src = temp_dir / "move_src.txt"
        dst = temp_dir / "move_dst.txt"
        src.write_text("move test")

        assert handler.move_file(str(src), str(dst))
        assert dst.exists()
        assert not src.exists()

    def test_delete_file(self, handler, temp_dir):
        """删除文件"""
        f = temp_dir / "delete_me.txt"
        f.write_text("to be deleted")

        assert handler.delete_file(str(f))
        assert not f.exists()

    def test_list_files(self, handler, temp_dir):
        """列出目录文件"""
        (temp_dir / "a.txt").write_text("a")
        (temp_dir / "b.txt").write_text("b")

        files = handler.list_files(str(temp_dir))
        assert "a.txt" in files
        assert "b.txt" in files

    def test_get_file_extension(self, handler):
        """获取文件扩展名"""
        assert handler.get_file_extension("test.json") == ".json"
        assert handler.get_file_extension("test.txt") == ".txt"
        assert handler.get_file_extension("archive.tar.gz") == ".gz"
        assert handler.get_file_extension("noext") == ""

    def test_get_filename_without_ext(self, handler):
        """获取不带扩展名的文件名"""
        assert handler.get_filename_without_ext("test.json") == "test"
        assert handler.get_filename_without_ext("report.txt") == "report"

    def test_get_latest_file(self, handler, temp_dir):
        """获取最新文件"""
        import time

        f1 = temp_dir / "old.txt"
        f2 = temp_dir / "new.txt"
        f1.write_text("old")
        time.sleep(0.1)
        f2.write_text("new")

        latest = handler.get_latest_file(str(temp_dir))
        assert latest is not None
        assert latest.name == "new.txt"
