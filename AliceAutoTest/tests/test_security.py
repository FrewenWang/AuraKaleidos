"""防止合并时把源工程的真实凭据重新带入版本库。"""

import re


def test_example_config_contains_no_webhook_token(project_root):
    config = (project_root / "config" / "config.example.ini").read_text(
        encoding="utf-8"
    )
    assert "access_token=" not in config
    assert not re.search(r"\b1\d{10}\b", config)


def test_python_sources_contain_no_real_dingtalk_token(project_root):
    token_pattern = re.compile(r"access_token=[0-9a-f]{32,}")
    for source in (project_root / "src").rglob("*.py"):
        assert not token_pattern.search(source.read_text(encoding="utf-8")), (
            source
        )


def test_protocol_constants_are_importable():
    from src.modules import px_constants

    assert px_constants.DEVICE_ANSWER_MACHINE_OK == 3000
    assert px_constants.MSG_OPERATION_OK == 1000
