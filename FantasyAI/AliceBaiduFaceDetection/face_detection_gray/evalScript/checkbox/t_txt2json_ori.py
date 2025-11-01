"""
将第三方提供的标注数据转化成json格式，并保存下来
Author: yubin
Date: 20211118
"""

import json
import os
import cv2 as cv
from PIL import Image
import numpy as np

def main(pathroot, path_labelfile, pathsave):
    """
    Args:
        pathroot: 原始图片路径
        path_labelfile: 标注文件，txt文本
        pathsave: 将txt转换成json之后的保存路径
    Returns:

    """
    allfiles = os.listdir(pathroot)
    for filet in allfiles:
        try:
            infos = open(pathlabel + filet + ".txt").readlines()[0]
            infos = infos.strip().split(",")
            pathImg = os.path.join(pathroot, filet)
            im = Image.open(pathImg)
            img = np.array(im, dtype=np.uint8)
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            boxitem = list(map(float, infos))
            l,t,r,b = boxitem
            name_new = filet[:-5]
            # cv.rectangle(img, (int(l),int(t)), (int(r), int(b)), (0,0,255), 2, 8, 0)
            # cv.imshow("img", img)
            # cv.waitKey(0)
            filedst = open(os.path.join(pathsave,  name_new + ".json"), 'w')
            dataRslt = {}
            dataRslt["WorkLoad"] = {}
            dataRslt["WorkLoad"]["Point Num"] = 0
            dataRslt["WorkLoad"]["scale_x"] = 1.
            dataRslt["DataList"] = []
            datalist0 = {}
            datalist0["type"] = "face_bbox"
            datalist0["id"] = "1"
            datalist0["coordinates"] = []
            dict_left = {"left": l}
            dict_top = {"top": t}
            dict_right = {"right": r}
            dict_bottom = {"bottom": b}
            datalist0["coordinates"].append(dict_left)
            datalist0["coordinates"].append(dict_top)
            datalist0["coordinates"].append(dict_right)
            datalist0["coordinates"].append(dict_bottom)

            dataRslt['DataList'].append(datalist0)
            filedst.write(json.dumps(dataRslt))

        except Exception as e:
            print (e)
            print (filet)

if __name__ == "__main__":


    pathimg = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/1000personFacialLandmark/标2_modify/image/"
    pathlabel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/1000personFacialLandmark/标2_modify/box/"
    pathsave = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/1000personFacialLandmark/标2_modify/json/"
    # 运行函数
    main(pathimg, pathlabel, pathsave)
