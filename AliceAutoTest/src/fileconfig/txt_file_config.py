"""
文本文件配置模块
向后兼容包装器，使用新的统一基类
"""

import os

from .file_config_base import TxtFileHandler


class TxtFileConfig(TxtFileHandler):
    """文本文件处理器（向后兼容）"""

    def __init__(self):
        super().__init__()
        self.basereportpath = str(self.get_report_path("reportpath"))

    @property
    def combination_path(self):
        """获取日志路径"""
        return os.path.join(self.basereportpath, r"log.txt")

    def wirte_file(self, wirte_data):
        """写入文件（向后兼容）"""
        return self.write_file(
            self.combination_path,
            str(self.datetime) + ":" + str(wirte_data) + "\n",
        )

    def read_file(self, filepath):
        """读取文件"""
        return super().read_file(filepath)

    def read_new_file(self):
        """读取目录下最新文件"""
        lists = os.listdir(self.basereportpath)
        lists.sort(
            key=lambda fn: os.path.getmtime(
                os.path.join(self.basereportpath, fn)
            )
        )
        return os.path.join(self.basereportpath, lists[-1])

    def copy_file(self, targetfile, sourcepath, targetpath):
        """复制文件"""
        files_list = os.listdir(self.config.get_filepath(sourcepath))
        if targetfile != "" and (targetfile in files_list):
            filename, filetype = targetfile.split(".")
            sp = os.path.join(self.config.get_filepath(sourcepath), targetfile)
            tp = os.path.join(self.config.get_filepath(targetpath), targetfile)
            super().copy_file(sp, tp)

    def rename_log(self):
        """重命名日志文件"""
        all_list = os.listdir(self.basereportpath)
        log_path = self.config.get_filepath("log_path")
        filepath, filename = os.path.split(log_path)
        if filename in all_list:
            file_name, filetype = filename.split(".")
            os.rename(
                log_path,
                os.path.join(
                    filepath, (file_name + self.datetime) + "." + filetype
                ),
            )
            return True
        return False

    def del_file(self, targetfile, filepath):
        """删除文件"""
        filepath = self.config.get_filepath(filepath)
        files_list = os.listdir(filepath)
        if targetfile != "" and (targetfile in files_list):
            os.remove(os.path.join(filepath, targetfile))
            return True
        else:
            for i in files_list:
                path = os.path.join(filepath, i)
                print(path)
                os.remove(path)
            return len(files_list)

    def create_folder(self):
        """创建文件夹"""
        import tempfile

        folder_path = os.path.join(
            tempfile.gettempdir(),
            "Python_auto",
            "CommonReport",
            "old_report",
            "one",
        )
        os.makedirs(folder_path, exist_ok=True)
