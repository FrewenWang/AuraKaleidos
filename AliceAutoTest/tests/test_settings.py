"""全局通知设置的本地配置与环境变量优先级测试。"""

import importlib

from src import settings


def test_notification_urls_use_local_config(
    tmp_path, monkeypatch, project_root
):
    local_config = tmp_path / "config.local.ini"
    local_config.write_text(
        "[DINGTALK]\n"
        "error_course = https://config.test/error\n"
        "start_course = https://config.test/start\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ALICE_AUTOTEST_CONFIG", str(local_config))
    monkeypatch.delenv("ALICE_AUTOTEST_ERROR_WEBHOOK", raising=False)
    monkeypatch.delenv("ALICE_AUTOTEST_START_WEBHOOK", raising=False)

    reloaded = importlib.reload(settings)

    assert reloaded.ERROR_COURSE == "https://config.test/error"
    assert reloaded.START_COURSE == "https://config.test/start"

    monkeypatch.setenv(
        "ALICE_AUTOTEST_CONFIG",
        str(project_root / "config" / "config.example.ini"),
    )
    importlib.reload(settings)


def test_notification_environment_has_priority(monkeypatch):
    monkeypatch.setenv("ALICE_AUTOTEST_START_WEBHOOK", "https://env.test/start")

    reloaded = importlib.reload(settings)

    assert reloaded.START_COURSE == "https://env.test/start"

    monkeypatch.delenv("ALICE_AUTOTEST_START_WEBHOOK")
    importlib.reload(settings)
