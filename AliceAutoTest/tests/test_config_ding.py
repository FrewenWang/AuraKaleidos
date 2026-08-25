"""通知配置合并与无网络降级测试。"""

import logging

from src.config import config_ding


class FakeConfig:
    values = {
        "release_pass_url": "https://config.test/pass",
        "release_error_url": "https://config.test/error",
        "test_pass_url": "https://config.test/test-pass",
        "test_error_url": "https://config.test/test-error",
        "error_course": "https://config.test/course",
        "user_right_info": "one, two",
        "user_error_info": "three",
    }

    def get_dingtalk(self, name):
        return self.values.get(name, "")


def _disable_file_logging(monkeypatch):
    logger = logging.getLogger("test-config-ding")
    monkeypatch.setattr(
        config_ding.ConfigLogging,
        "write_logging",
        lambda _self: logger,
    )


def test_config_file_fallback(monkeypatch):
    _disable_file_logging(monkeypatch)
    ding = config_ding.ConfigDing(config=FakeConfig())

    assert ding.release_pass_url == "https://config.test/pass"
    assert ding.user_right_info == ["one", "two"]
    assert ding.user_error_info == ["three"]


def test_legacy_environment_has_priority(monkeypatch):
    _disable_file_logging(monkeypatch)
    monkeypatch.setenv(
        "ALICE_AUTOTEST_DING_RELEASE_PASS", "https://env.test/pass"
    )
    monkeypatch.setenv("ALICE_AUTOTEST_DING_SUCCESS_USERS", " a, ,b ")

    ding = config_ding.ConfigDing(config=FakeConfig())

    assert ding.release_pass_url == "https://env.test/pass"
    assert ding.user_right_info == ["a", "b"]


def test_missing_webhook_never_calls_network(monkeypatch):
    _disable_file_logging(monkeypatch)
    monkeypatch.setattr(
        config_ding.requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("不应发送网络请求")
        ),
    )
    ding = config_ding.ConfigDing(config=FakeConfig())

    assert ding.base_request("", {"msgtype": "text"}) is False
