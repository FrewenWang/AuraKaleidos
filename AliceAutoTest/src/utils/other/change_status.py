"""
状态变更模块 - 处理课件状态更新
"""

import time

import requests

from src.config.config_logging import ConfigLogging


class ChangeStatus:
    """状态变更处理器"""

    def __init__(self):
        self.logger = ConfigLogging().write_logging()

    def fun_request(self, content: dict) -> bool:
        """发送状态变更请求"""
        try:
            url = "http://phoenix.100tal.com/courseCheck/action/autoCheck/syncCoursewareCheckResult"
            headers = {
                "Content-Type": "application/json",
                "T-px-Validate-Token": "ss",
            }

            data = {
                "coursewareId": content.get("classid", ""),
                "checkStatus": 1,  # 待测试状态
                "checkBeginDate": int(time.time() * 1000),
                "checkEndDate": int(time.time() * 1000),
            }

            response = requests.post(url, headers=headers, json=data, timeout=6)
            result = response.json()

            if result.get("code") == 0:
                self.logger.info(f"状态变更成功: {content.get('classid')}")
                return True
            else:
                self.logger.error(f"状态变更失败: {result}")
                return False
        except Exception as e:
            self.logger.error(f"状态变更异常: {e}")
            return False


# 向后兼容的别名
Change_Status = ChangeStatus
