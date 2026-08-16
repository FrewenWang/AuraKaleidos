#!/usr/local/bin/python
# @Time : 2019/9/20 23:18
# @Author : liuqi
# @File : jietu1.py
# @Software: PyCharm

import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import psutil
from psutil import Process, pids

from src.config.base_config import ReadBaseConfig
from src.config.config_ding import ConfigDing

# PyQt5 和 cv2 为可选依赖，仅在截图功能需要时导入
try:
    from PyQt5.QtWidgets import QApplication

    _PYQT5_AVAILABLE = True
except ImportError:
    _PYQT5_AVAILABLE = False

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


class ConfigScreen:
    def __init__(self):
        # 使用跨平台路径
        project_root = Path(__file__).parent.parent.parent.parent
        self.basepath = str(project_root / "assets" / "ScreenImage")
        self._copy_image_path = str(project_root / "assets" / "OldScreenImage")
        if not os.path.exists(self._copy_image_path):
            os.makedirs(self._copy_image_path)
        self.ding = ConfigDing()
        self.p = None
        self.base_config_data = ReadBaseConfig()
        self.telnet = self.base_config_data.get_login("telnet")

    def get_pid(self, pid_name):
        for pid in pids():
            try:
                self.p = Process(pid)
            except psutil.NoSuchProcess:
                continue
            if self.p.name() == pid_name:
                return pid_name

    def createtime(self):
        local_time = time.localtime(time.time())
        nowtime = time.strftime("%Y-%m-%d_%H%M%S", local_time)
        return nowtime

    # 截屏函数
    def screenshot(self):
        if not _PYQT5_AVAILABLE:
            print("⚠️  PyQt5未安装，截图功能不可用")
            return
        QApplication(sys.argv)
        screen = QApplication.primaryScreen()
        pix = screen.grabWindow(QApplication.desktop().winId())
        pix.save(os.path.join(self.basepath, self.createtime() + ".jpg"))

    # 匹配截屏图片
    def match_image(self, image_path1, image_path2):
        if not _CV2_AVAILABLE:
            print("⚠️  opencv未安装，图片匹配功能不可用")
            return False
        image_path1 = os.path.join(self.basepath, image_path1)
        image_path2 = os.path.join(self.basepath, image_path2)
        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)
        difference = cv2.subtract(image1, image2)
        result = not np.any(difference)
        if result is True:
            print("前后两张图片一样")
            return True
        else:
            print("前后两张图片不一样")
            return False

    def create_match(self):
        count = 0
        image_lists = os.listdir(self.basepath)
        if len(image_lists) > 1:
            for i in range(len(image_lists)):
                if image_lists[i] != image_lists[-1] and count <= 5:
                    print(image_lists[i], image_lists[i + 1])
                    if self.match_image(image_lists[i], image_lists[i + 1]):
                        count += 1
                    else:
                        for j in range(0, i + 1):
                            shutil.move(
                                os.path.join(self.basepath, image_lists[j]),
                                os.path.join(
                                    self._copy_image_path, image_lists[j]
                                ),
                            )
                        count = 0
                    print("count:", count)
                elif count > 5:
                    print("测试不通过")
                    return True
        else:
            print("图片数量少于2个")

    # 运行截屏
    def _screen(self, screentime):
        from src.config.platform_compat import is_windows

        pid_name = "Phoenix.exe" if is_windows() else "Phoenix"
        while self.get_pid(pid_name):
            self.screenshot()  # 截屏
            if self.create_match():  # 匹配
                content = (
                    "auto当前"
                    + "连续超过10次图片对比一致，请查询卡顿原因！！！"
                    + "向日葵远程地址："
                    + str(self.telnet)
                )
                self.ding.ding_test(3, content)
                break
            time.sleep(screentime)
        print("当前学生端未启动！！！")

    # 备份文件
    def _copy_screen_image(self, targetpath):
        if not os.path.exists(targetpath):
            os.makedirs(targetpath)
            return
        files = os.listdir(targetpath)
        if files:
            for file in files:
                if os.path.isfile(os.path.join(targetpath, file)):
                    shutil.move(
                        os.path.join(targetpath, file),
                        os.path.join(self._copy_image_path, file),
                    )

    # 删除文件
    def del_screen_image(self):
        nowtime = int(time.time())
        for root, _dirs, filenames in os.walk(self._copy_image_path):
            for filename in filenames:
                src_file = os.path.join(root, filename)
                if int(os.path.getctime(src_file)) + 172800 < nowtime:
                    os.remove(src_file)

    def run_screen(self):
        if not os.path.isdir(self._copy_image_path):
            os.makedirs(self._copy_image_path)
        if not os.path.isdir(self.basepath):
            os.makedirs(self.basepath)
        self._copy_screen_image(self.basepath)  # 清空截图目录文件
        self._screen(10)


if __name__ == "__main__":
    data = ConfigScreen()
    data.run_screen()
