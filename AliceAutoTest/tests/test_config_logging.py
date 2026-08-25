"""日志编码与 handler 生命周期测试。"""

import logging

from src.config.config_logging import ConfigLogging


def _reset_logger():
    logger = logging.getLogger("logging.log")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    return logger


def test_repeated_initialization_does_not_duplicate_handlers(
    tmp_path, monkeypatch
):
    logger = _reset_logger()
    monkeypatch.setenv("ALICE_AUTOTEST_REPORT_PATH_LOGGING_PATH", str(tmp_path))

    first = ConfigLogging().write_logging()
    second = ConfigLogging().write_logging()

    assert first is second is logger
    assert len(logger.handlers) == 2
    _reset_logger()


def test_log_file_is_utf8(tmp_path, monkeypatch):
    _reset_logger()
    monkeypatch.setenv("ALICE_AUTOTEST_REPORT_PATH_LOGGING_PATH", str(tmp_path))
    logger = ConfigLogging().write_logging()

    logger.info("中文日志")
    for handler in logger.handlers:
        handler.flush()

    content = (tmp_path / "logging.log").read_text(encoding="utf-8")
    assert "中文日志" in content
    _reset_logger()
