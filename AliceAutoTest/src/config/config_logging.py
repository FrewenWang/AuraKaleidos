#!/usr/local/bin/python
# @Time : 2019/8/2 23:57
# @Author : liuqi
# @File : config_logging.py
# @Software: PyCharm
import logging
import os
import shutil
import time
from pathlib import Path

from src.config.base_config import ReadBaseConfig
from src.config.datetime_utils import ConfigTime

"""
    日志级别NOTSET < DEBUG < INFO < WARNING < ERROR < CRITICAL
"""


class ConfigLogging:
    back_logging_path = str(Path.home() / "logs" / "back_logging")
    mv_logging_path = str(Path.home() / "logs" / "px_logging")

    def __init__(self):
        self.logging_name = "logging.log"
        self.time_info = ConfigTime().reporttime  # 获取当前时间
        self.report_path = ReadBaseConfig().get_reportpath(
            "logging_path"
        )  # 获取存储日志路径
        self.src_path = os.path.join(self.report_path, self.logging_name)
        self.logger = logging.getLogger(self.logging_name)  # logging方法初始化
        self.logger.setLevel(level=logging.DEBUG)  # 设置全局日志等级
        # self.write_logging()

    # 记录日志文件
    def config_logging(self):
        Path(self.report_path).mkdir(parents=True, exist_ok=True)
        if self.logger.handlers:
            return

        file_handler = logging.FileHandler(
            self.src_path, mode="a", encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.ERROR)

        #  格式化记录日志内容
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s - [%(lineno)s]\n >>%(message)s"
        )
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        self.logger.propagate = False

    def write_logging(self):
        self.config_logging()
        return self.logger

    # 备份日志文件
    def copy_logging(self):
        if (
            os.path.isfile(self.src_path)
            and os.path.getsize(self.src_path) != 0
        ):
            back_dir = Path(self.report_path) / "back_logging"
            back_dir.mkdir(parents=True, exist_ok=True)
            target_path = str(back_dir / f"{self.time_info}.log")
            shutil.copy(self.src_path, target_path)
            if os.path.getsize(target_path) == os.path.getsize(self.src_path):
                with open(self.src_path, "r+") as read_src_file:
                    read_src_file.truncate()  # 清空文件内容
        else:
            print("日志路径不存在")

    def mv_logging(self):
        if not os.path.exists(self.mv_logging_path):
            os.makedirs(self.mv_logging_path)
        nowtime = int(time.time())
        for root, _dirs, filenames in os.walk(self.back_logging_path):
            for filename in filenames:
                src_file = os.path.join(root, filename)
                drc_file = os.path.join(self.mv_logging_path, filename)
                if int(os.path.getctime(src_file)) + 86400 < nowtime:
                    shutil.move(src_file, drc_file)

    def handle_log(self):
        """
        备份当前日志和历史日志
        :return:
        """
        self.mv_logging()
        self.copy_logging()


if __name__ == "__main__":
    # data = ConfigLogging().write_logging()
    # client_info = {'code': 0, 'hint': 'success', 'msg': '成功', 'result': {'success': False}}
    # data.info("COMPONENTS_INIT_READY" + str(client_info))
    # from test import Test
    # data1 = Test()
    # data1.runlogging()
    data = ConfigLogging()
    data.mv_logging()
