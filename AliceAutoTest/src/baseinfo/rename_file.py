# -*- conding:utf-8 -*-
# @Time : 2019/3/8 11:03
# @Author : liuqi
# @File : ReadFile.py
# @Software: PyCharm
import os


# 操作文件
class HandleFile:
    def __init__(self):
        self.monitor = (
            r"C:\Users\phoenix\AppData\Roaming\phoenix\bin\Monitor.exe"
        )
        self.px_answerer = r"C:\Users\phoenix\AppData\Roaming\phoenix\bin\pxanswer\PxAnswerer.exe"

    # 修改文件名字
    def handlefile(self, srcpath):
        (filepath, tempfilename) = os.path.split(srcpath)
        (filename, extension) = os.path.splitext(tempfilename)
        newfilename = filename + "2"
        newtempfilename = newfilename + extension
        dstpath = os.path.join(filepath, newtempfilename)
        try:
            os.rename(srcpath, dstpath)
        except FileNotFoundError:
            return False

    def renamefile(self):
        self.handlefile(self.monitor)
        self.handlefile(self.px_answerer)


# 主程序
if __name__ == "__main__":
    data = HandleFile()
    data.renamefile()
