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
    infos = open(path_labelfile).readlines()
    infos = infos[1:]
    print (len(infos))
    nid = 0
    for linet in infos:
        try:
            linet1 = linet.strip().split("\t")
            pathmid1 = linet1[1]
            pathImg = os.path.join(pathroot, pathmid1)

            # pathImg = pathImg.replace(" ", "")
            # print (pathImg)
            # img = cv.imread(pathImg)

            im = Image.open(pathImg)
            img = np.array(im, dtype=np.uint8)
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

            boxinfos = linet1[3]
            boxinfos = json.loads(boxinfos)
            result = boxinfos["result"]
            allBoxes = []
            for result_item in result:
                elements = result_item["elements"]
                for elements_item in elements:
                    points = elements_item["points"]
                    pt_x1 = points[0]["x"]
                    pt_y1 = points[0]["y"]
                    pt_x2 = points[1]["x"]
                    pt_y2 = points[1]["y"]

                    x1,y1,x2,y2 = map(int, [pt_x1, pt_y1, pt_x2, pt_y2])
                    # cv.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 3, 8, 0)
                    x1, y1, x2, y2 = map(float, [pt_x1, pt_y1, pt_x2, pt_y2])
                    allBoxes.append([x1,y1, x2, y2])

            # print(img.shape)
            # cv.imshow("img", img)
            # cv.waitKey(0)
            nid += 1
            l,t,r,b = allBoxes[0]
            name_new = str(nid).zfill(7)

            filedst = open(os.path.join(pathsave,  name_new + ".json"), 'w')
            pathsaveImg = os.path.join(pathsave,  name_new + ".jpg")
            dataRslt = {}
            dataRslt["WorkLoad"] = {}
            dataRslt["WorkLoad"]["Point Num"] = 0
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
            cv.imwrite(pathsaveImg, img)
        except Exception as e:
            print (e)
            print (linet)

if __name__ == "__main__":

    # 原始图片路径
    pathOriImg = "D:/face_detect/data/OMS_Face_后排小人脸_20211025_数量20000"
    # 交付的标注信息
    path_labelFile = "D:/face_detect/data/人脸标框-第一批-10849帧数据交付.txt"
    # 保存路径， 转换后的json文件
    pathsave_ImgJson = "D:/face_detect/data/data_biaozhu_1"


    # 原始图片路径
    pathOriImg = "/media/baidu/ssd1/标注数据/OMS_Face_后排小人脸_20211025_数量20000"
    # 交付的标注信息
    path_labelFile = "/media/baidu/ssd1/标注数据/人脸标框-第一批-10849帧数据交付.txt"
    # 保存路径， 转换后的json文件
    pathsave_ImgJson = "/media/baidu/ssd1/标注数据/second/first/orisize"

    # 运行函数
    main(pathOriImg, path_labelFile, pathsave_ImgJson)
