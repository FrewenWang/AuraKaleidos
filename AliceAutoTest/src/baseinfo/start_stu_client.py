#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import wmi


class StartStuClient:
    # 调用win10客户端的runpcandca.bat文件启用客户端，输入密码为北京地区的账号和密码
    def start_sut_client(self):
        conn = wmi.WMI()
        try:
            filename = r"C:\Users\phoenix\AppData\Roaming\phoenix\runpcandca.bat"  # 此文件在远程服务器上
            cmd_callbat = rf"cmd /c call {filename}"
            conn.Win32_Process.Create(
                CommandLine=cmd_callbat
            )  # 执行bat文件   Win32_Process.Create
        except Exception as e:
            print("启动学生端出错，具体error内容：", e)


if __name__ == "__main__":
    data = StartStuClient()
    data.start_sut_client()
