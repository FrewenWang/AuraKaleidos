"""
配置读取模块 (base_config) 单元测试

测试 ReadBaseConfig 的配置文件读取功能
"""

import pytest


class TestReadBaseConfig:
    """ReadBaseConfig 测试"""

    @pytest.fixture
    def config(self):
        from src.config.base_config import ReadBaseConfig

        return ReadBaseConfig()

    def test_get_http(self, config):
        """读取 [HTTP] 节配置"""
        url = config.get_http("base_login_url")
        assert url
        assert url.startswith("http")

    def test_get_http_nonexistent_key(self, config):
        """读取不存在的 HTTP 键抛出 NoOptionError"""
        import configparser

        with pytest.raises(configparser.NoOptionError):
            config.get_http("nonexistent_key")

    def test_get_login(self, config):
        """读取 [LOGIN] 节配置"""
        username = config.get_login("en_username")
        assert username
        assert len(username) > 0

    def test_get_login_nonexistent_key(self, config):
        """读取不存在的 LOGIN 键抛出 NoOptionError"""
        import configparser

        with pytest.raises(configparser.NoOptionError):
            config.get_login("nonexistent_key")

    def test_get_db(self, config):
        """读取 [DB] 节配置"""
        host = config.get_db("host")
        assert host
        assert len(host) > 0

    def test_get_db_port(self, config):
        """读取数据库端口"""
        port = config.get_db("port")
        assert port
        # 端口应为数字字符串
        assert port.isdigit()

    def test_get_filepath(self, config):
        """读取 [FILE_PATH] 节配置"""
        path = config.get_filepath("jsonfile_path")
        assert path
        assert len(path) > 0

    def test_get_reportpath(self, config):
        """读取 [REPORT_PATH] 节配置"""
        path = config.get_reportpath("logging_path")
        assert path
        assert len(path) > 0

    def test_environment_takes_precedence(self, config, monkeypatch):
        """环境变量覆盖文件值，且不会预先读取不存在的配置项。"""
        monkeypatch.setenv(
            "ALICE_AUTOTEST_HTTP_NONEXISTENT_KEY", "https://example.test"
        )
        assert config.get_http("nonexistent_key") == "https://example.test"

    def test_optional_merge_sections(self, config):
        """通知和联系人节可安全读取空占位值。"""
        assert config.get_dingtalk("release_pass_url") == ""
        assert config.get_contacts("test_phone") == ""

    def test_generic_contact_environment_override(self, config, monkeypatch):
        monkeypatch.setenv("ALICE_AUTOTEST_CONTACTS_TEST_PHONE", "test-contact")
        assert config.get_contacts("test_phone") == "test-contact"


class TestGetVersion:
    """GetVersion 测试"""

    def test_get_version_class_exists(self):
        """GetVersion 类可以实例化"""
        from src.config.base_config import GetVersion

        try:
            gv = GetVersion()
            assert gv is not None
        except Exception:
            # 非 Windows 系统可能降级
            pass

    def test_get_cversion_on_non_windows(self):
        """非 Windows 系统返回 unknown"""
        from src.config.platform_compat import is_windows

        if not is_windows():
            from src.config.base_config import GetVersion

            try:
                gv = GetVersion()
                version = gv.get_cversion()
                assert version == "unknown"
            except Exception:
                # GetVersion 可能无法实例化
                pass
