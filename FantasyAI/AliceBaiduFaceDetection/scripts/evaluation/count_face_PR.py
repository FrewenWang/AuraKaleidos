"""
Authors:
Date:
"""
import matplotlib.pyplot as plt
import numpy as np
import os

def getIou(box1, box2):
    boxIOU = -1.0
    l1,t1,r1,b1 = box1
    l2,t2,r2,b2 = box2[0]

    l = max(l1,l2)
    t = max(t1,t2)
    r = min(r1,r2)
    b = min(b1,b2)
    if r<=l or b<=t:
        return boxIOU
    else:
        area1 = (r1-l1)*(b1-t1)
        area2 = (r2-l2)*(b2-t2)
        area_inter = (r-l)*(b-t)
        boxIOU = area_inter/(area1 + area2 - area_inter)
        return boxIOU

def getBox(filebox):
    box = []
    infos = open(filebox).readlines()
    for linet in infos:
        linet = linet.strip().split("\t")
        label, l,t,r,b = linet
        l,t,r,b = map(float, [l,t,r,b])
        box.append([l,t,r,b])
    return box

def main_score_threshold(pathLabel, pathrslt, score_threshold=0.3):
    patherror = pathrslt + "_error"
    if not os.path.isdir(patherror):
        os.makedirs(patherror)

    allImgNums = 0.
    true_face = 0
    error_face = 0
    allIous = []
    allFiles = os.listdir(pathrslt)
    for filet in allFiles:
        if "txt" in filet :
            continue

        allImgNums += 1.
        # fileinfo = filet.split(".")[0]
        # filetxt = fileinfo + ".txt"
        filetxt = filet
        infos = open(os.path.join(pathrslt, filetxt + ".txt")).readlines()
        box_pre = []
        for linet in infos:
            # label, score, l,t,w,h = linet.strip().split("\t")
            score, l, t, r, b = linet.strip().split("\t")
            score = float(score)
            l,t,r,b = float(l), float(t), float(r), float(b)
            # r = l+w
            # b = t+h

            if score < score_threshold:
                continue
            box_pre.append([l,t,r,b])
        fileinfo = filet.split(".")
        filetxt1 = ".".join(fileinfo[:-1])
        filetxt = filetxt1 + ".txt"
        # box_label = getLabel(os.path.join(pathLabel, filexml))
        box_label = getBox(os.path.join(pathLabel, filetxt))

        for boxItem in box_pre:
            iou_score = getIou(boxItem, box_label)
            allIous.append(iou_score)
            if iou_score < 0.5:
                error_face += 1.0
                # shutil.copy(os.path.join(pathrslt, filet), os.path.join(patherror, filet))
            else:
                true_face += 1.0
    print ("all img nums:", allImgNums)
    print ("recall:", true_face, true_face/(allImgNums))
    print ("precision:", true_face/(true_face + error_face + 1e-6))
    print ("mIOU:", sum(allIous)/(len(allIous) + 1e-6), len(allIous))
    print ("*"*200)

    return true_face/(allImgNums), true_face/(true_face + error_face + 1e-6),  sum(allIous)/(len(allIous) + 1e-6)

if __name__ == "__main__":
    pathLabel = "/root/paddlejob/data_train/val_resize/boxes"
    pathrslt = "/root/paddlejob/data_train/val_resize/image_drawrslt"

    all_scores = np.linspace(0, 100, 11)
    all_recalls = []
    all_precisions = []
    all_mious = []
    print(len(all_scores), all_scores)
    for scoretmp in all_scores:
        scoretmp = scoretmp / 100.
        recall_item, precision_item, mIou_item = main_score_threshold(pathLabel, pathrslt, scoretmp)

        all_recalls.append(round(recall_item, 5))
        all_precisions.append(round(precision_item, 5))
        all_mious.append(mIou_item)

    print("++" * 100)
    print("all_scores:", all_scores)
    print("all_recalls:", all_recalls)
    print("all_precisions:", all_precisions)
    print("all_mious:", all_mious)
    print("++" * 100)

    precision_map = [all_precisions[0]]
    for i in range(1, len(all_precisions)):
        if all_precisions[i] >= max(precision_map):
            precision_map.append(all_precisions[i])
        else:
            precision_map.append(max(precision_map))
    mAP_value = sum(precision_map) / len(precision_map)
    mAP_value = round(mAP_value, 4)
    print("mAP", mAP_value)
    # plt.title("")
    plt.plot(all_recalls, all_precisions, label="PR_mAP=" + str(mAP_value))
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.grid()
    plt.legend()
    # plt.savefig("pr_curver.jpg")
    plt.savefig("/root/paddlejob/workspace/log/pr_curver.jpg")
    # plt.show()