# -*- conding:utf-8 -*-
# @Time : 2019/4/13 18:21
# @Author : liuqi
# @File : dingding.py
# @Software: PyCharm
import os

import requests

from src.config.base_config import ReadBaseConfig
from src.config.config_logging import ConfigLogging


def _split_list(value):
    """将逗号分隔的配置转换为去空值的列表。"""
    return [item.strip() for item in value.split(",") if item.strip()]


class ConfigDing:
    def __init__(self, config=None):
        self.logger = ConfigLogging().write_logging()
        self._config = config or ReadBaseConfig()
        self.release_pass_url = self._setting(
            "release_pass_url", "ALICE_AUTOTEST_DING_RELEASE_PASS"
        )
        self.release_error_url = self._setting(
            "release_error_url", "ALICE_AUTOTEST_DING_RELEASE_ERROR"
        )
        self.test_pass_url = self._setting(
            "test_pass_url", "ALICE_AUTOTEST_DING_TEST_PASS"
        )
        self.test_error_url = self._setting(
            "test_error_url", "ALICE_AUTOTEST_DING_TEST_ERROR"
        )
        self.error_course = self._setting(
            "error_course", "ALICE_AUTOTEST_DING_ERROR_COURSE"
        )
        self.timeout = 6
        self.headers = {"Content-Type": "application/json"}
        self.user_right_info = self._setting_list(
            "user_right_info", "ALICE_AUTOTEST_DING_SUCCESS_USERS"
        )
        self.user_error_info = self._setting_list(
            "user_error_info", "ALICE_AUTOTEST_DING_ERROR_USERS"
        )

    def _setting(self, config_name, legacy_env_name):
        env_value = os.getenv(legacy_env_name)
        if env_value is not None:
            return env_value
        return self._config.get_dingtalk(config_name)

    def _setting_list(self, config_name, legacy_env_name):
        env_value = os.getenv(legacy_env_name)
        if env_value is not None:
            return _split_list(env_value)
        return _split_list(self._config.get_dingtalk(config_name))

    # 钉钉机器人请求数据
    def base_request(self, *args):
        # try:
        # rq = requests.post(url=url, data=json.dumps(content), headers=self.headers, timeout=self.timeout)
        if not args or not args[0]:
            self.logger.warning("未配置钉钉 webhook，跳过消息发送")
            return False
        try:
            rq = requests.post(
                args[0],
                json=args[1],
                headers=self.headers,
                timeout=self.timeout,
            )
            print(rq.text)
            if not rq:
                self.logger.debug("ding:" + str(rq))
            return bool(rq)
        except requests.RequestException as e:
            self.logger.error("ding_error:" + str(e))
            return False

    # markdown数据组装
    def base_content(self, content):
        content_info = {}
        if content:
            # content_info= {
            # 'msgtype': 'markdown',
            # "markdown": {
            #     "title": "每日自动化测试课件状态报告",
            #     "text": content
            # },
            # "at": {
            #     "atMobiles": self.user_error_info,
            #     "isAtAll": False
            # }
            # }
            content_info = {
                "msgtype": "markdown",
                "markdown": {
                    "title": content["title"],
                    "text": "### "
                    + content["title"]
                    + "\n"
                    + "#### 【测试结果信息】"
                    + "\n"
                    +
                    # "- 测试结论：" + content["report"] + "\n" +
                    "- 测试结论："
                    + content["result_name"]
                    + "\n"
                    + "- 异常信息："
                    + content["errorReason"]
                    + "\n"
                    + "- 测试时长："
                    + content["used_time"]
                    + "\n"
                    + "***\n"
                    + "#### 【远程访问地址】"
                    + "\n"
                    +
                    # "- 测试主机向日葵ID：" + content["SunflowerName"] + "\n" +
                    "- 测试主机向日葵ID："
                    + content["remote_id"]
                    + "\n"
                    +
                    # "- 测试主机向日葵密码：" + content["SunflowerPwd"] + "\n" +
                    "- 测试主机向日葵密码："
                    + content["remote_pwd"]
                    + "\n"
                    + "***\n"
                    + "#### 【测试环境】"
                    + "\n"
                    + "- 测试主机名称："
                    + content["hostname"]
                    + "\n"
                    + "- 测试主机MAC："
                    + content["mac"]
                    + "\n"
                    + "- 测试账号："
                    + content["username"]
                    + "\n"
                    + "- 学生端版本："
                    + content["px_version"]
                    + "\n"
                    + "***"
                    + "\n"
                    + "##### 【课件信息】"
                    + "\n"
                    + "- 测试课件名称："
                    + content["classname"]
                    + "\n"
                    + "- 课件LAST_CID："
                    + content["cid"]
                    + "\n"
                    + "- 课件运行组件统计："
                    + str(content["componentcount"])
                    + "\n",
                },
                "at": {"atMobiles": self.user_error_info, "isAtAll": False},
            }
        return content_info

    # @人员
    def at_user(self):
        at_user = {
            "msgtype": "text",
            "text": {"content": "auto请查阅以上课件校验报告##"},
            "at": {"atMobiles": "", "isAtAll": False},
        }
        return at_user

    def _ding_info_release(self, status, content):
        at_user = self.at_user()
        if content and status != 1:
            content_info = self.base_content(content)
            print(content_info)
            if status == 0:
                at_user["at"]["atMobiles"] = self.user_right_info
                self.base_request(self.release_pass_url, content_info)
                self.base_request(self.release_pass_url, at_user)
            else:
                at_user["text"]["content"] = content["errorReason"]
                at_user["at"]["atMobiles"] = self.user_error_info
                self.base_request(self.release_error_url, content_info)
                self.base_request(self.release_error_url, at_user)
        elif status == 1:
            at_user["text"]["content"] = content["image_screen_info"]
            at_user["at"]["atMobiles"] = self.user_error_info
            self.base_request(self.release_error_url, at_user)

    def _ding_info_test(self, state, content):
        """
        :param state: 1:测试通过 2:测试不通过 3:课前检测异常
        :param content:
        :return:
        """
        at_user = self.at_user()
        if content and state == 3:
            at_user["text"]["content"] = content
            at_user["at"]["atMobiles"] = self.user_error_info
            self.base_request(self.test_error_url, at_user)

        else:
            try:
                content_info = self.base_content(content)  # markdown格式
                print(content_info)
            except Exception as e:
                content_info = e
            if state == 1:
                at_user["at"]["atMobiles"] = self.user_right_info
                print(content_info)
                print(at_user)
                self.base_request(self.test_pass_url, content_info)
                self.base_request(self.test_pass_url, at_user)
            else:
                at_user["at"]["atMobiles"] = self.user_error_info
                self.base_request(self.test_error_url, content_info)
                self.base_request(self.test_error_url, at_user)

    def ding_release(self, status, content):
        self._ding_info_release(status, content)
        self._ding_info_test(status, content)

    def ding_test(self, state, content):
        self._ding_info_test(state, content)
