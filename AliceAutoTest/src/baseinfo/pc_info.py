#!/usr/bin/env python
# @Time    : 2020/1/8 15:14
# @Author  : liuqi
# @Site    :
# @File    : pc_info.py
# @Software: PyCharm
import os
import re
import time

from src.config.config_hardware import HardwareConfig, PidConfig
from src.config.platform_compat import get_platform, is_windows


class BasePcInfo:
    def __init__(self):
        if is_windows():
            self.filepath = os.path.join(
                rf"C:\Users\{HardwareConfig.get_user()}",
                r"AppData\Local\Temp\AIBrowser\logs",
            )
        else:
            self.filepath = os.path.join(
                get_platform().get_temp_path(), "AIBrowser", "logs"
            )

    def _cut_out(self, start, end, string):
        result = re.findall(f".*{start}(.*){end}.*", string)
        data = "{" + result[0] + "}"
        return eval(data)

    # 获取最新JavaScript.log日志文件
    def _read_path(self):
        files = os.listdir(self.filepath)
        dir_list = []
        logfile = None
        for file in files:
            if "JavaScript" in file:
                dir_list.append(file)
        dir_list = sorted(
            dir_list,
            key=lambda x: os.path.getmtime(os.path.join(self.filepath, x)),
        )
        try:
            logfile = os.path.join(self.filepath, dir_list[-1])
        except IndexError as e:
            print("_read_path_erro:" + str(e))
        return logfile

    # 读取最新日志文件获取deviceId信息
    def read_file(self):
        result_data = None
        try:
            with open(self._read_path(), encoding="UTF-8") as log_file:
                lines = list(log_file)
        except Exception as e:
            print("读取最新日志文件获取deviceId信息:" + str(e))
            lines = []
        listqrcodes = []
        if lines:
            for file in lines:
                if (
                    "checkQrCode" in file
                    and "deviceId" in file
                    and "ajax" in file
                    and "postId" in file
                ):
                    listqrcodes.append(file)
        else:
            print("获取二维码日志文件异常：" + str(lines))
        try:
            if listqrcodes:
                result = eval(repr(listqrcodes[-1]).replace(r"\\", ""))
                result_data = result.split("=")
            else:
                print("listqrcode_异常:", listqrcodes)
        except ImportError as e:
            print("get_loginfo_error:" + str(e))
        # return self._cut_out(a, b, result_data[1])
        return len(listqrcodes), result_data

    @classmethod
    def get_pc_deviceid(cls):
        result = None
        count = 0
        a = "{"
        b = "},{"
        process_name = PidConfig().get_pid("Phoenix.exe")
        if process_name:
            while count < 3:
                result = cls().read_file()
                if result[1]:
                    count += 3
                else:
                    time.sleep(5)
                    count += 1
            result = cls()._cut_out(a, b, result[1][1])
        else:
            print("当前学生端未启动！！！")
        return result


if __name__ == "__main__":
    data = BasePcInfo.get_pc_deviceid()
    print(data)
