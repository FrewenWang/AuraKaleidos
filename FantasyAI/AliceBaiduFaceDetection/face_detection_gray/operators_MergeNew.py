import numpy as np
import xml.etree.ElementTree as et
import os
import cv2 as cv
import random
import time
import copy


def get_label_box(pathxml):
    # path = "/media/baidu/ssd2/ppyolo/6w_data/baidu_hand_xml/gesture/fist/12_00010.xml"
    info = et.parse(pathxml)
    root = info.getroot()
    obj = root.find("object")
    bndbox = obj.find("bndbox")
    xmin = bndbox.find("xmin")
    xmax = bndbox.find("xmax")
    ymin = bndbox.find("ymin")
    ymax = bndbox.find("ymax")

    l, t, r, b = map(float, [xmin.text, ymin.text, xmax.text, ymax.text])
    l, t, r, b = map(int, [l, t, r, b])
    return [l, t, r, b]

def getPartImg_Neg(img):
    partNegImg = None
    h,w,_ = img.shape

    return partNegImg, []

'''
以手为中心，截取手的图片
'''
def getPartImg_Pos(img, box, expandratio=6):
    img_hand1, box_new = None, []

    h1, w1, _ = img.shape
    l1, t1, r1, b1 = box
    box1_h1, box1_w1 = b1 - t1, r1 - l1
    centx1, centy1 = (l1 + r1) / 2., (t1 + b1) / 2.
    fscale1 = random.uniform(1, expandratio)
    # print("edge scale, 1-4:", fscale1)
    box1_h1 = box1_h1 * fscale1
    box1_w1 = box1_w1 * fscale1
    l1_new = max(0, centx1 - box1_w1 / 2.)
    t1_new = max(0, centy1 - box1_h1 / 2.)
    r1_new = min(centx1 + box1_w1 / 2., w1)
    b1_new = min(centy1 + box1_h1 / 2., h1)

    l1_new, t1_new, r1_new, b1_new = map(int, [l1_new, t1_new, r1_new, b1_new])
    box1_h1_new = b1_new - t1_new
    box1_w1_new = r1_new - l1_new
    # print("w:", box1_w1_new, "h:", box1_h1_new)
    img_hand1 = img[t1_new:b1_new, l1_new:r1_new, :]

    l, t, r, b = box
    l = l - l1_new
    r = r - l1_new
    t = t - t1_new
    b = b - t1_new
    box_new.append([l, t, r, b])

    # l,t,r,b = map(int, [l,t,r,b])
    # cv.rectangle(img_hand1, (l,t), (r,b), (0,0,255), 5, 8, 0)
    # cv.imshow("img_hand1_part", img_hand1)
    # cv.imshow("img_hand1", img)
    # cv.waitKey(0)
    return img_hand1, box_new[0]


def resize_partHand(img, box, win_w=640, win_h=360):
    l, t, r, b = box
    # l,t,r,b = map(int, [l,t,r,b])
    # cv.rectangle(img, (l,t), (r,b), (0,0,255), 5, 8, 0)
    # cv.imshow("test_1", img)
    # cv.waitKey(0)

    h, w, _ = img.shape
    if w > win_w or h > win_h:
        scaleW = float(w) / win_w
        scaleH = float(h) / win_h
        fscale = max(scaleW, scaleH)
        h_dst = int(h / fscale)
        w_dst = int(w / fscale)
        img2 = cv.resize(img, (w_dst, h_dst))
        l, t, r, b = l / fscale, t / fscale, r / fscale, b / fscale
        # l,t,r,b = map(int, [l,t,r,b])
        # cv.rectangle(img2, (l,t), (r,b), (0,0,255), 3, 8, 0)
        # cv.imshow("img2_resize", img2)
        # cv.waitKey(0)
        return img2, [l, t, r, b]

    return img, box


def mergeImages(allImgs, allBoxes, allNegs=[]):
    # imgDst = np.ones((720, 1280, 3), dtype=np.uint8)
    try:
        rand_neg_id = random.randint(0, len(allNegs)-1)
        imgDst = cv.imread(allNegs[rand_neg_id], 0)
        imgDst = cv.cvtColor(imgDst, cv.COLOR_BGR2RGB)
        imgDst = cv.resize(imgDst, (1280, 720))
    except:
        imgDst = np.ones((720, 1280, 1), dtype=np.uint8)
    boxDst = []
    # 123.675, 116.28, 103.53
    # imgDst[:, :, 0] = 123
    # imgDst[:, :, 1] = 116
    # imgDst[:, :, 2] = 103
    num_imgs = len(allImgs)
    # img1, img2, img3, img4, img5, img6, img7, img8, img9 = allImgs
    # box1, box2, box3, box4, box5, box6, box7, box8, box9 = allBoxes
    '''
    2种情况:
    1：4个人脸
    2：5个人脸    
    '''

    allBoxes1 = copy.deepcopy(allBoxes)
    allBoxes = []
    for item in allBoxes1:
        allBoxes.append(list(item[0]))

    fProb_Neg = 0.5
    fscale_condition = random.uniform(0, 1)
    if fscale_condition < 0.2:  # 情况1 两行两列
        imgTmp = []
        boxTmp = []
        for i in range(4):
            if random.uniform(0, 1) < fProb_Neg:
                imgTmp.append(None)
                boxTmp.append(None)
            else:
                id_item = random.randint(0, num_imgs - 1)
                img_hand_item, box_hand_item = getPartImg_Pos(allImgs[id_item], allBoxes[id_item])
                imgTmp.append(img_hand_item)
                boxTmp.append(box_hand_item)

        # 4只手，每只手的最大宽高640,360，两行两列
        win_w = 640
        win_h = 360
        imgTmp2 = []
        boxTmp2 = []
        for i in range(4):
            if imgTmp[i] is None or boxTmp[i] is None:
                imgTmp2.append(None)
                boxTmp2.append(None)
            else:
                img_hand_item2, box_hand_item2 = resize_partHand(imgTmp[i], boxTmp[i], win_w, win_h)
                imgTmp2.append(img_hand_item2)
                boxTmp2.append(box_hand_item2)

        for row in range(2):
            for col in range(2):
                id_win = row * 2 + col
                if imgTmp2[id_win] is None or boxTmp2[id_win] is None:
                    continue
                img_hand_tmp = imgTmp2[id_win]
                l_start = win_w * col + random.randint(0, win_w - img_hand_tmp.shape[1])
                t_start = win_h * row + random.randint(0, win_h - img_hand_tmp.shape[0])
                imgDst[t_start: t_start + img_hand_tmp.shape[0], l_start:l_start + img_hand_tmp.shape[1], :] = img_hand_tmp

                l, t, r, b = boxTmp2[id_win]
                l, t, r, b = l + l_start, t + t_start, r + l_start, b + t_start
                boxDst.append([l, t, r, b])

        # for id in range(4):
        #     l, t, r, b = map(int, boxDst[id])
        #     cv.rectangle(imgDst, (l, t), (r, b), (0, 0, 255), 3, 8, 0)
        # cv.imshow("imgDst", imgDst)
        # img2 = cv.resize(imgDst, (320, 160))
        # cv.imshow('img2', img2)
        # cv.waitKey(0)

    elif fscale_condition < 0.35 :  # 情况2， 两行三列
        imgTmp = []
        boxTmp = []
        for i in range(6):
            if random.uniform(0, 1) < fProb_Neg:
                imgTmp.append(None)
                boxTmp.append(None)
            else:
                id_item = random.randint(0, num_imgs - 1)
                img_hand_item, box_hand_item = getPartImg_Pos(allImgs[id_item], allBoxes[id_item])
                imgTmp.append(img_hand_item)
                boxTmp.append(box_hand_item)

        # 6只手，每只手的最大宽高426,360，两行三列
        win_w = 426
        win_h = 360
        imgTmp2 = []
        boxTmp2 = []
        for i in range(6):
            if imgTmp[i] is None or boxTmp is None:
                imgTmp2.append(None)
                boxTmp2.append(None)
            else:
                img_hand_item2, box_hand_item2 = resize_partHand(imgTmp[i], boxTmp[i], win_w, win_h)
                imgTmp2.append(img_hand_item2)
                boxTmp2.append(box_hand_item2)

        for row in range(2):
            for col in range(3):
                id_win = row * 3 + col
                if imgTmp2[id_win] is None or boxTmp2[id_win] is None:
                    continue
                img_hand_tmp = imgTmp2[id_win]
                l_start = win_w * col + random.randint(0, win_w - img_hand_tmp.shape[1])
                t_start = win_h * row + random.randint(0, win_h - img_hand_tmp.shape[0])
                imgDst[t_start: t_start + img_hand_tmp.shape[0], l_start:l_start + img_hand_tmp.shape[1],
                :] = img_hand_tmp

                l, t, r, b = boxTmp2[id_win]
                l, t, r, b = l + l_start, t + t_start, r + l_start, b + t_start
                boxDst.append([l, t, r, b])

        # for id in range(6):
        #     l, t, r, b = map(int, boxDst[id])
        #     cv.rectangle(imgDst, (l, t), (r, b), (0, 0, 255), 3, 8, 0)
        # cv.imshow("imgDst", imgDst)
        # img2 = cv.resize(imgDst, (320, 160))
        # cv.imshow('img2', img2)
        # cv.waitKey(0)

    elif fscale_condition < 0.5:  # 情况3，三行两列
        imgTmp = []
        boxTmp = []
        for i in range(6):
            if random.uniform(0, 1) < fProb_Neg:
                imgTmp.append(None)
                boxTmp.append(None)
            else:
                id_item = random.randint(0, num_imgs - 1)
                img_hand_item, box_hand_item = getPartImg_Pos(allImgs[id_item], allBoxes[id_item])
                imgTmp.append(img_hand_item)
                boxTmp.append(box_hand_item)

        # 6只手，每只手的最大宽高640,240，三行两列
        win_w = 640
        win_h = 240
        imgTmp2 = []
        boxTmp2 = []
        for i in range(6):
            if imgTmp[i] is None or boxTmp[i] is None:
                imgTmp2.append(None)
                boxTmp2.append(None)
            else:
                img_hand_item2, box_hand_item2 = resize_partHand(imgTmp[i], boxTmp[i], win_w, win_h)
                imgTmp2.append(img_hand_item2)
                boxTmp2.append(box_hand_item2)

        for row in range(3):
            for col in range(2):
                id_win = row * 2 + col
                if imgTmp2[id_win] is None or boxTmp2[id_win] is None:
                    continue
                img_hand_tmp = imgTmp2[id_win]
                l_start = win_w * col + random.randint(0, win_w - img_hand_tmp.shape[1])
                t_start = win_h * row + random.randint(0, win_h - img_hand_tmp.shape[0])
                imgDst[t_start: t_start + img_hand_tmp.shape[0], l_start:l_start + img_hand_tmp.shape[1],
                :] = img_hand_tmp

                l, t, r, b = boxTmp2[id_win]
                l, t, r, b = l + l_start, t + t_start, r + l_start, b + t_start
                boxDst.append([l, t, r, b])

        # for id in range(6):
        #     l, t, r, b = map(int, boxDst[id])
        #     cv.rectangle(imgDst, (l, t), (r, b), (0, 0, 255), 3, 8, 0)
        # cv.imshow("imgDst", imgDst)
        # img2 = cv.resize(imgDst, (320, 160))
        # cv.imshow('img2', img2)
        # cv.waitKey(0)

    elif fscale_condition < 0.7:  # 情况4 两行四列
        imgTmp = []
        boxTmp = []
        for i in range(8):
            if random.uniform(0, 1) < fProb_Neg:
                imgTmp.append(None)
                boxTmp.append(None)
            else:
                id_item = random.randint(0, num_imgs - 1)
                img_hand_item, box_hand_item = getPartImg_Pos(allImgs[id_item], allBoxes[id_item])
                imgTmp.append(img_hand_item)
                boxTmp.append(box_hand_item)

        # 8只手，每只手的最大宽高320,360，两行四列
        win_w = 320
        win_h = 360
        imgTmp2 = []
        boxTmp2 = []
        for i in range(8):
            if imgTmp[i] is None or boxTmp[i] is None:
                imgTmp2.append(None)
                boxTmp2.append(None)
            else:
                img_hand_item2, box_hand_item2 = resize_partHand(imgTmp[i], boxTmp[i], win_w, win_h)
                imgTmp2.append(img_hand_item2)
                boxTmp2.append(box_hand_item2)

        for row in range(2):
            for col in range(4):
                id_win = row * 4 + col
                if imgTmp2[id_win] is None or boxTmp2[id_win] is None:
                    continue
                img_hand_tmp = imgTmp2[id_win]
                l_start = win_w * col + random.randint(0, win_w - img_hand_tmp.shape[1])
                t_start = win_h * row + random.randint(0, win_h - img_hand_tmp.shape[0])
                imgDst[t_start: t_start + img_hand_tmp.shape[0], l_start:l_start + img_hand_tmp.shape[1],
                :] = img_hand_tmp

                l, t, r, b = boxTmp2[id_win]
                l, t, r, b = l + l_start, t + t_start, r + l_start, b + t_start
                boxDst.append([l, t, r, b])

        # for id in range(8):
        #     l, t, r, b = map(int, boxDst[id])
        #     cv.rectangle(imgDst, (l, t), (r, b), (0, 0, 255), 3, 8, 0)
        # cv.imshow("imgDst", imgDst)
        # img2 = cv.resize(imgDst, (320, 160))
        # cv.imshow('img2', img2)
        # cv.waitKey(0)

    elif fscale_condition < 0.8:  # 情况5 四行两列
        imgTmp = []
        boxTmp = []
        for i in range(8):
            if random.uniform(0, 1) < fProb_Neg:
                imgTmp.append(None)
                boxTmp.append(None)
            else:
                id_item = random.randint(0, num_imgs - 1)
                img_hand_item, box_hand_item = getPartImg_Pos(allImgs[id_item], allBoxes[id_item], expandratio=3)
                imgTmp.append(img_hand_item)
                boxTmp.append(box_hand_item)

        # 8只手，每只手的最大宽高320,360，四行两列
        win_w = 640
        win_h = 180
        imgTmp2 = []
        boxTmp2 = []
        for i in range(8):
            if imgTmp[i] is None or boxTmp[i] is None:
                imgTmp2.append(None)
                boxTmp2.append(None)
            else:
                img_hand_item2, box_hand_item2 = resize_partHand(imgTmp[i], boxTmp[i], win_w, win_h)
                imgTmp2.append(img_hand_item2)
                boxTmp2.append(box_hand_item2)

        for row in range(4):
            for col in range(2):
                id_win = row * 2 + col
                if imgTmp2[id_win] is None or boxTmp2[id_win] is None:
                    continue
                img_hand_tmp = imgTmp2[id_win]
                l_start = win_w * col + random.randint(0, win_w - img_hand_tmp.shape[1])
                t_start = win_h * row + random.randint(0, win_h - img_hand_tmp.shape[0])
                imgDst[t_start: t_start + img_hand_tmp.shape[0], l_start:l_start + img_hand_tmp.shape[1],
                :] = img_hand_tmp

                l, t, r, b = boxTmp2[id_win]
                l, t, r, b = l + l_start, t + t_start, r + l_start, b + t_start
                boxDst.append([l, t, r, b])

        # for id in range(8):
        #     l, t, r, b = map(int, boxDst[id])
        #     cv.rectangle(imgDst, (l, t), (r, b), (0, 0, 255), 3, 8, 0)
        # cv.imshow("imgDst", imgDst)
        # img2 = cv.resize(imgDst, (320, 160))
        # cv.imshow('img2', img2)
        # cv.waitKey(0)

    elif fscale_condition < 0.9:  # 情况6 三行三列
        imgTmp = []
        boxTmp = []
        for i in range(9):
            if random.uniform(0, 1) < fProb_Neg:
                imgTmp.append(None)
                boxTmp.append(None)
            else:
                id_item = random.randint(0, num_imgs - 1)
                img_hand_item, box_hand_item = getPartImg_Pos(allImgs[id_item], allBoxes[id_item], expandratio=3)
                imgTmp.append(img_hand_item)
                boxTmp.append(box_hand_item)

        # 9只手，每只手的最大宽高426,240，三行三列
        win_w = 426
        win_h = 240
        imgTmp2 = []
        boxTmp2 = []
        for i in range(9):
            if imgTmp[i] is None or boxTmp[i] is None:
                imgTmp2.append(None)
                boxTmp2.append(None)
            else:
                img_hand_item2, box_hand_item2 = resize_partHand(imgTmp[i], boxTmp[i], win_w, win_h)
                imgTmp2.append(img_hand_item2)
                boxTmp2.append(box_hand_item2)

        for row in range(3):
            for col in range(3):
                id_win = row * 3 + col
                if imgTmp2[id_win] is None or boxTmp2[id_win] is None:
                    continue
                img_hand_tmp = imgTmp2[id_win]
                l_start = win_w * col + random.randint(0, win_w - img_hand_tmp.shape[1])
                t_start = win_h * row + random.randint(0, win_h - img_hand_tmp.shape[0])
                imgDst[t_start: t_start + img_hand_tmp.shape[0], l_start:l_start + img_hand_tmp.shape[1],
                :] = img_hand_tmp

                l, t, r, b = boxTmp2[id_win]
                l, t, r, b = l + l_start, t + t_start, r + l_start, b + t_start
                boxDst.append([l, t, r, b])

        # for id in range(9):
        #     l, t, r, b = map(int, boxDst[id])
        #     cv.rectangle(imgDst, (l, t), (r, b), (0, 0, 255), 3, 8, 0)
        # cv.imshow("imgDst", imgDst)
        # img2 = cv.resize(imgDst, (320, 160))
        # cv.imshow('img2', img2)
        # cv.waitKey(0)
    else:  # 情况7  四行四列
        imgTmp = []
        boxTmp = []
        for i in range(16):
            if random.uniform(0, 1) < fProb_Neg:
                imgTmp.append(None)
                boxTmp.append(None)
            else:
                id_item = random.randint(0, num_imgs - 1)
                img_hand_item, box_hand_item = getPartImg_Pos(allImgs[id_item], allBoxes[id_item], expandratio=3)
                imgTmp.append(img_hand_item)
                boxTmp.append(box_hand_item)

        # 16只手，每只手的最大宽高320,180，四行四列
        win_w = 320
        win_h = 180
        imgTmp2 = []
        boxTmp2 = []
        for i in range(16):
            if imgTmp[i] is None or boxTmp[i] is None:
                imgTmp2.append(None)
                boxTmp2.append(None)
            else:
                img_hand_item2, box_hand_item2 = resize_partHand(imgTmp[i], boxTmp[i], win_w, win_h)
                imgTmp2.append(img_hand_item2)
                boxTmp2.append(box_hand_item2)

        for row in range(4):
            for col in range(4):
                id_win = row * 4 + col
                if imgTmp2[id_win] is None or boxTmp2[id_win] is None:
                    continue
                img_hand_tmp = imgTmp2[id_win]
                l_start = win_w * col + random.randint(0, win_w - img_hand_tmp.shape[1])
                t_start = win_h * row + random.randint(0, win_h - img_hand_tmp.shape[0])
                imgDst[t_start: t_start + img_hand_tmp.shape[0], l_start:l_start + img_hand_tmp.shape[1],
                :] = img_hand_tmp

                l, t, r, b = boxTmp2[id_win]
                l, t, r, b = l + l_start, t + t_start, r + l_start, b + t_start
                boxDst.append([l, t, r, b])

        # for id in range(16):
        #     l, t, r, b = map(int, boxDst[id])
        #     cv.rectangle(imgDst, (l, t), (r, b), (0, 0, 255), 3, 8, 0)
        # cv.imshow("imgDst", imgDst)
        # img2 = cv.resize(imgDst, (320, 160))
        # cv.imshow('img2', img2)
        # cv.waitKey(0)
    # print ("*"*100)
    # print (boxDst)
    # for item in boxDst:
    #     l,t,r,b = map(int, item)
    #     cv.rectangle(imgDst, (l,t), (r,b), (0,0,255), 1, 8, 0)
    # cv.imwrite("/home/baidu/Desktop/t1/" + str(int(1000*(time.time()))) + ".jpg", imgDst[:,:,::-1])
    if len(boxDst) == 0:
        boxDst = [allBoxes[0]]
        imgDst = allImgs[0]
    boxDst = np.array(boxDst).astype(np.float32)
    return imgDst, boxDst


def main():
    pathImg = "/media/baidu/3.6TB_SSD/val_data/valImg"
    pathXml = "/media/baidu/3.6TB_SSD/val_data/valXml"
    pathsave = "/home/baidu/Desktop/model_img_tmp2"
    pathNeg = "/media/baidu/ssd2/ppyolo/opendata/coco_train2014_noHand"
    allNegs1 = os.listdir(pathNeg)
    allNegs = []
    for filet in allNegs1:
        allNegs.append(os.path.join(pathNeg, filet))

    allfiles = os.listdir(pathImg)
    allxmls = os.listdir(pathXml)
    num_files = len(allfiles)
    random.seed(100)
    random.shuffle(allfiles)

    for nid in range(len(allfiles)):

        # for filet in allfiles:
        print("nid", nid)
        filet = allfiles[nid]
        img = cv.imread(os.path.join(pathImg, filet))
        filexml = filet.split(".")
        filexml = ".".join(filexml[:-1]) + ".xml"
        box = get_label_box(os.path.join(pathXml, filexml))

        allPosImgs = []
        allPosBoxes = []
        allPosImgs.append(img)
        allPosBoxes.append(box)
        for i in range(15):
            id1 = random.randint(0, num_files - 1)
            filet1 = allfiles[id1]
            img1 = cv.imread(os.path.join(pathImg, filet1))
            filexml1 = filet1.split(".")
            filexml1 = ".".join(filexml1[:-1]) + ".xml"
            box1 = get_label_box(os.path.join(pathXml, filexml1))
            allPosImgs.append(img1)
            allPosBoxes.append(box1)

        imgNew, boxNew = mergeImages(allPosImgs, allPosBoxes, allNegs)

        for id in range(len(boxNew)):
            l, t, r, b = map(int, boxNew[id])
            cv.rectangle(imgNew, (l, t), (r, b), (0, 0, 255), 3, 8, 0)

        if len(boxNew) == 0:
            boxNew = [allPosBoxes[0]]
            imgNew = allPosImgs[0]
            for item in boxNew:
                l,t,r,b = map(int, item)
                cv.rectangle(imgNew, (l,t), (r,b), (0,0,0), 5, 8, 0)

        cv.imshow("imgDst", imgNew)
        img2 = cv.resize(imgNew, (320, 160))
        cv.imshow('img2', img2)
        intTime = int(1000*(time.time()))
        cv.imwrite(os.path.join(pathsave, str(intTime)+".jpg"), imgNew)
        cv.waitKey(1)

if __name__ == "__main__":
    main()
