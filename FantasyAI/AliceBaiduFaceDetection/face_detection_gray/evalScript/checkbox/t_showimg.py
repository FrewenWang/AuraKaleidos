import os
import cv2 as cv

def main():
    pathimg = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/1000personFacialLandmark/标2_modify/image/"
    pathbox = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/1000personFacialLandmark/标2_modify/box/"

    allfiles = os.listdir(pathimg)
    print (len(allfiles))
    for filet in allfiles:
        infos = open(pathbox + filet + ".txt").readlines()[0]
        box = infos.strip().split(",")
        box = list(map(float, box))
        l,t,r,b = list(map(int, box))
        img = cv.imread(pathimg + filet)

        cv.rectangle(img, (l,t), (r,b), (0,0,255), 2, 8, 0)
        cv.imshow("img", img)
        cv.waitKey(0)

if __name__ == "__main__":
    # main()
    import os
    import shutil
    pathimg = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/1000personFacialLandmark/标2_modify/image/"
    pathjson = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/1000personFacialLandmark/标2_modify/json/"

    allfiles = os.listdir(pathjson)
    for filet in allfiles:
        shutil.copy(pathjson + filet, pathimg + filet)