import copy
import os
import random
import cv2
import numpy as np
import json
from dataset import DetDataset
import cv2 as cv


def get_box_point(pathJson):
    # label_infos = {}
    # label_infos['facebox'] = []
    # label_infos['facepoints'] = []

    try:
        infos = json.loads(open(pathJson).read())
        WorkLoad = infos['WorkLoad']
        fscale = WorkLoad["scale_x"]
        DataList = infos["DataList"]
        Point_num = WorkLoad['Point Num']
        # 目前Point_num 有三个数值，2, 106, 72,
        # 暂时先设置一个阈值5,大于等于5则使用人脸点推理方框，否则直接读取face_bbox

        if Point_num >= 5:
            # print ("****", pathJson)
            all_x = []
            all_y = []
            for item in DataList:
                if item['type'] != "Point":
                    continue
                fx, fy = item['coordinates']
                # label_infos['facepoints'].append([fx, fy])
                all_x.append(fx)
                all_y.append(fy)
            left = min(all_x)
            right = max(all_x)
            top = min(all_y)
            bottom = max(all_y)
        else:
            print (pathJson)
            for item in DataList:
                if item['type'] == "face_bbox":
                    coordinates = item['coordinates']
                    left = coordinates[0]['left']
                    top = coordinates[1]['top']
                    right = coordinates[2]['right']
                    bottom = coordinates[3]['bottom']
                    # label_infos['facebox'] = [left, top, right, bottom]
        if right - left < 10 or bottom - top < 10:
            return []
        return [left * fscale, top * fscale, right * fscale, bottom * fscale]
    except:
        # print ("pathJson error:", pathJson)
        return []

    # return label_infos


def getallfiles(path):
    allfiles = []
    for root, folders, files in os.walk(path):
        for filet in files:
            if filet.split(".")[-1] in ['jpg', 'jpeg', 'png', 'bmp', 'JPG', 'JPEG', 'BMP', 'PNG']:
                allfiles.append(os.path.join(root, filet))
    return allfiles


def parse_dataset(pathroot):
    allfiles = getallfiles(pathroot)
    print ("filenums:", len(allfiles))
    random.seed(100)
    random.shuffle(allfiles)
    for filet in allfiles:
        pathimg1 = filet
        # 根据img生成json，二者在同一路径下，只是文件后缀不一样
        filepathInfo = pathimg1.split("/")
        imgname = filepathInfo[-1]
        imgname1 = imgname.split(".")[:-1]
        imgname2 = ".".join(imgname1)
        filepathPrefix = "/".join(filepathInfo[:-1])
        # 该路径下没有人脸点标注信息
        if "/1000personFacialLandmark/标2" in pathimg1 or "oms_SmallFace_biaozhu" in pathimg1:
            jsonname = imgname2 + ".json"
        else:
            # 有人脸点标注信息
            jsonname = "out_" + imgname2 + ".json"

        pathimg = pathimg1
        img = cv.imread(pathimg, 0)

        pathjson = os.path.join(filepathPrefix, jsonname)
        box_label = get_box_point(pathjson)

        l,t,r,b = list(map(int, box_label))
        cv.rectangle(img, (l,t), (r,b), (0,255,0), 3, 8, 0)
        cv.imshow("img", img)
        cv.waitKey(0)

if __name__ == "__main__":
    #此路径需重新生成方框 43000张
    pathroot = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/1000personFacialLandmark/标2"

    #
    pathroot = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/人脸"
    pathroot ="/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/1000personFacialLandmark/标2_modify/image"
    parse_dataset(pathroot)
