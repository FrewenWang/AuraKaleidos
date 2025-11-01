"""
********
Authors:
Date:
"""
import json
import os
import random
import cv2 as cv
import copy

def check_box():
    """
    检查标签是否正确
    """
    pathroot = "/media/baidu/ssd1/标注数据/norm_smallSize/oms_SmallFace_biaozhu1_20211115/"

    allFiles = os.listdir(pathroot)
    random.seed(100)
    random.shuffle(allFiles)
    for filet in allFiles:
        if ".json" in filet:
            continue
        img = cv.imread(pathroot + filet)
        filejson = json.loads(open(pathroot + filet[:-4] + ".json").read())
        WorkLoad = filejson["WorkLoad"]
        DataList = filejson["DataList"][0]
        coordinates = DataList["coordinates"]
        l = coordinates[0]["left"]
        t = coordinates[1]["top"]
        r = coordinates[2]["right"]
        b = coordinates[3]["bottom"]
        scale = WorkLoad["scale_x"]

        l,t,r,b = l*scale,t*scale,r*scale,b*scale
        l,t,r,b = map(int, [l,t,r,b])
        cv.rectangle(img, (l,t), (r,b), (0,0,255), 5, 8, 0)
        print (img.shape, "w=",r-l, "h=",b-t)
        cv.imshow("img", img)
        cv.waitKey(0)


def resize_img_(pathOri, pathDst):
    allFiles = os.listdir(pathOri)
    for filet in allFiles:
        filename_prefix = filet[:-4]
        if "json" in filet:
            continue

        norm_h = 360
        norm_w = 640
        img = cv.imread(os.path.join(pathOri,  filet))
        img_h, img_w, _ = img.shape

        filejson = filet.replace(".jpg", ".json")
        infos = json.loads(open(os.path.join(pathOri,  filejson)).read())
        infos_bak = copy.deepcopy(infos)
        WorkLoad = infos_bak["WorkLoad"]
        if img_h < norm_h or img_w < norm_w:
            WorkLoad["scale_x"] = 1.
            WorkLoad["scale_y"] = 1.
            imgdst = img
        else:
            fscale_x = 1.0 * norm_w / img_w
            fscale_y = 1.0 * norm_h / img_h
            fscale = max(fscale_x, fscale_y)
            WorkLoad["scale_x"] = fscale
            WorkLoad["scale_y"] = fscale
            h_dst = int(img_h * fscale + 0.5)
            w_dst = int(img_w * fscale + 0.5)
            imgdst = cv.resize(img, (w_dst, h_dst))

        cv.imwrite(os.path.join(pathDst,  filet), imgdst)
        filedst = open(os.path.join(pathDst, filename_prefix + ".json"), "w")
        infos["WorkLoad"] = WorkLoad
        filedst.write(json.dumps(infos))


if __name__ == "__main__":
    pathOri = "D:/face_detect/data/data_biaozhu_1"
    pathDst = "D:/face_detect/data/data_biaozhu_1_resize"

    pathOri = "/media/baidu/ssd1/标注数据/second/first/orisize"
    pathDst = "/media/baidu/ssd1/标注数据/second/first/normsize"

    resize_img_(pathOri, pathDst)

