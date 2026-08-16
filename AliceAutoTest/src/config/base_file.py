#!/usr/bin/env python
# @Time    : 2019/10/16 16:29
# @Author  : liuqi
# @Site    :
# @File    : base_file.py
# @Software: PyCharm
import getpass
import os
import shutil

from src.config.base_config import ReadBaseConfig
from src.config.platform_compat import get_platform, is_windows


class BaseFile:
    def __init__(self):
        self.base_info = ReadBaseConfig()
        self._platform = get_platform()
        if is_windows():
            self.resource_path = os.path.join(
                rf"C:\Users\{getpass.getuser()}",
                r"AppData\Roaming\phoenix\Resource",
            )
            self.component_path = os.path.join(
                rf"C:\Users\{getpass.getuser()}",
                self.base_info.get_filepath("component_path"),
            )
        else:
            # macOS/Linux: 使用跨平台路径
            app_data = self._platform.get_app_data_path()
            self.resource_path = os.path.join(app_data, "Resource")
            self.component_path = os.path.join(
                app_data, self.base_info.get_filepath("component_path")
            )

    def _del_dir_tree(self, resource_path):
        """递归删除目录及其子目录,　子文件"""
        try:
            for path in os.listdir(resource_path):
                del_path = os.path.join(resource_path, path)
                if os.path.isdir(del_path):
                    shutil.rmtree(del_path)
                else:
                    os.remove(del_path)
        except Exception:
            print("component路径不存在")

    # 删除路径下所有数据
    def del_resource(self):
        self._del_dir_tree(self.resource_path)

    # 删除组件路径下所有数据
    def del_component(self):
        self._del_dir_tree(self.component_path)

    # 删除pid记录数据
    def del_pid(self):
        pid_path = self._get_pid_path()
        if os.path.exists(pid_path):
            os.remove(pid_path)

    # 修改PID文件
    def update_pid(self, number):
        pid_path = self._get_pid_path()
        # 确保目录存在
        os.makedirs(os.path.dirname(pid_path), exist_ok=True)
        with open(pid_path, "w") as f:
            f.write(number)

    def _get_pid_path(self):
        """获取PID文件路径（跨平台）"""
        if is_windows():
            return r"C:\run\run_phoenix.pid"
        else:
            return os.path.join(
                self._platform.get_temp_path(), "run_phoenix.pid"
            )

    # 清除所有课件相关数据
    def all_data(self):
        self.del_resource()
        self.del_component()
        self.del_pid()


if __name__ == "__main__":
    data = BaseFile()
    print(data.component_path)
    print(data.resource_path)
