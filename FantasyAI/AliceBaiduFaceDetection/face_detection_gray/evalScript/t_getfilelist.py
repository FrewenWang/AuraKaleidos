"""
生成文件名称列表
Authors:
Date:
"""
import os
import random
import datetime

def main(pathimg, pathtrain, pathval, foldername, folderPrefix):
    """
    Args:
        pathimg: 图片路径
        pathtxt: 文本文件路径
        folderPrefix: 文件夹的前缀
    Returns:

    """
    allFiles = os.listdir(pathimg)
    random.seed(100)
    random.shuffle(allFiles)

    valNum = int(0.05 * len(allFiles))
    filelist_train = open(os.path.join(pathtrain, "train_" + foldername + ".txt"), "w")
    for filet in  allFiles[valNum:]:
        if ".jpg" not in filet:
            continue
        filelist_train.write(folderPrefix + filet + "\n")
    filelist_val = open(os.path.join(pathval, "val_" + foldername + ".txt"), "w")
    for filet in  allFiles[:valNum]:
        if ".jpg" not in filet:
            continue
        filelist_val.write(folderPrefix + filet + "\n")

if __name__ == "__main__":
    foldername = "oms_SmallFace_biaozhu_20211119"
    folder_prefix = "./2021Q1Q2zhongce/"+ foldername + "/"
    pathImg = "/media/baidu/ssd1/标注数据/second/first/folder_root/2021Q1Q2zhongce/" + foldername
    pathTxt = "/media/baidu/ssd1/标注数据/second/first/folder_root/filelist"
    pathtrain = os.path.join(pathTxt, "train_filelist")
    if not os.path.isfile(pathtrain):
        os.makedirs(pathtrain)
    pathval = os.path.join(pathTxt, "val_filelist")
    if not os.path.isfile(pathval):
        os.makedirs(pathval)

    main(pathImg, pathtrain, pathval, foldername, folder_prefix)
