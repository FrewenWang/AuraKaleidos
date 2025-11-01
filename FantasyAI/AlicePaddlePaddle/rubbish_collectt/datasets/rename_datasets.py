from distutils import filelist
from importlib.resources import path
import os


def rename():
    res = os.listdir('../')
    for img in res:
        i = 0
        flag = os.path.isdir(img)
        if (flag == False):
            continue
        path = img
        filelist = os.listdir(path)
        for file in filelist:
            i = i + 1
            OldDir = os.path.join(path, file)
            if os.path.isdir(OldDir):
                continue
            fileName = os.path.splitext(file)[0]
            fileType = os.path.splitext(file)[1]
            newDir = os.path.join(path, str(i) + fileType)
            os.rename(OldDir, newDir)


rename()
