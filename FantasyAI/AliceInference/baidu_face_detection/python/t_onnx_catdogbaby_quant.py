# encoding:utf-8
import os
import numpy as np
import onnxruntime
import cv2
import math


def getallfiles(path):
    allFiles = []
    for root, folder, files in os.walk(path):
        for filet in files:
            if filet.split(".")[-1] in ["jpg", "jpeg", "png", "bmp", "JPG", "JPEG", "PNG", "BMP"]:
                allFiles.append(os.path.join(root, filet))
    return allFiles


def predict_images_onnx(ort_sess):
    """
    :param net:
    :param path:
    :return:
    """
    image = cv2.imread("demo_dog.jpg")

    img_w = image.shape[1]
    img_h = image.shape[0]

    # print('original image shape:', image.shape)
    image_disp = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_origin = cv2.resize(image, (288, 160))
    image_pre = (image_origin.astype(np.float32) / 255. - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    # print('resized image shape:', image_pre.shape)
    image = np.expand_dims(image_pre, axis=0)

    image = (image.astype(np.float32))
    image = np.transpose(image, [0, 3, 1, 2])

    ort_inputs = {ort_sess.get_inputs()[0].name: image}

    head_output = ort_sess.run(None, ort_inputs)

    num_cls = 3  # 猫，狗，婴儿
    num_lvls = 3
    grid_hs = [5, 10, 20]
    grid_ws = [9, 18, 36]
    grid_ss = [1., 1., 1.]
    num_bias = 3
    conf_thr_table = [0.6, 0.45, 0.3]  # conf_thr_table = [0.005, 0.005, 0.005]
    det_w = 288
    det_h = 160
    box_rslt = []  # 存放结果
    for c in range(num_cls):
        bboxes = []
        scoreIdx = []  # [[float, int]]
        bboxIdx = 0
        for i in range(num_lvls):
            gridH = grid_hs[i]
            gridW = grid_ws[i]
            fpStride = grid_ss[i]

            # anStride = (num_cls + 5) * fpStride
            outData1 = head_output[i]
            outData1 = np.transpose(outData1, [0, 2, 3, 1])
            outData = np.reshape(outData1, [-1])

            for yid in range(gridH):
                for xid in range(gridW):
                    for b in range(num_bias):
                        # obj_idx = b * anStride + 4 * fpStride + yid * gridW + xid
                        obj_idx = yid * gridW * num_bias + xid * num_bias + b
                        score = CalConfScore(outData, obj_idx, fpStride, c)
                        if score >= conf_thr_table[c]:
                            boxtmp = DecodeBBox(outData, obj_idx, fpStride, i,
                                                xid, yid, det_w, det_h, img_w, img_h, b,
                                                grid_ws, grid_hs, bboxes)
                            bboxes.append(boxtmp + [score])
                            # bboxes.append(score)
                            scoreIdx.append([score, bboxIdx])
                            bboxIdx += 1

        # sorted(scoreIdx, ), #分数从大到小
        def takefirst(elem):
            return -elem[0]

        scoreIdx.sort(key=takefirst)
        indices = NMS(scoreIdx, bboxes)
        for k in range(len(indices)):
            xmin = min(max(bboxes[indices[k]][0] / img_w, 0.), 1.)
            ymin = min(max(bboxes[indices[k]][1] / img_h, 0.), 1.)
            xmax = min(max(bboxes[indices[k]][2] / img_w, 0.), 1.)
            ymax = min(max(bboxes[indices[k]][3] / img_h, 0.), 1.)
            if xmax <= xmin or ymax <= ymin:
                continue
            result = []
            result.append(c)
            result.append(bboxes[indices[k]][4])
            result.append(xmin)
            result.append(ymin)
            result.append(xmax)
            result.append(ymax)
            box_rslt.append(result)
    # print ("box_rslt:", box_rslt)

    # 此函数用于画图显示
    for i, pred in enumerate(box_rslt):
        # if pred[1] >= -0.2:
        l, t, r, b = box_rslt[i][2:]
        # centx = (l + r) / 2.
        # centy = (t + b) / 2.
        # h_pre = b-t
        # w_pre  = r-l
        # w_pre = 0.58*w_pre
        # l = max(0,  centx - w_pre/2.)
        # r = min(centx + w_pre/2., image_disp.shape[1])
        l, t, r, b = l * img_w, t * img_h, r * img_w, b * img_h
        l, t, r, b = map(int, [l, t, r, b])
        if pred[0] == 0:
            cv2.rectangle(image_disp, (l, t), (r, b), (255, 0, 0), 2, 8, 0)  # 猫
        elif pred[0] == 1:
            cv2.rectangle(image_disp, (l, t), (r, b), (0, 255, 0), 2, 8, 0)  # 狗
        elif pred[0] == 2:
            cv2.rectangle(image_disp, (l, t), (r, b), (0, 0, 255), 2, 8, 0)  # 婴儿
        fscore = pred[1]
        cv2.putText(image_disp, str(box_rslt[i][0]) + "-" + str(round(fscore, 3)), (l, t), 1, 1, (0, 0, 255), 1, 8, 0)
    print("pet baby check:", box_rslt)
    cv2.imshow('preds', image_disp)
    cv2.waitKey(0)


def Sigmoid(x):
    y = 1. / (1. + math.exp(-x))
    return y


def CalConfScore(pred, idx, stride, cls_id):
    # objectness = Sigmoid(pred[idx])
    # confidence = Sigmoid(pred[idx + (1+cls_id)*int(stride)])

    objectness1 = (pred[idx * 8 + 4])
    confidence1 = (pred[idx * 8 + 4 + (1 + cls_id) * int(stride)])
    objectness = Sigmoid(objectness1)
    confidence = Sigmoid(confidence1)
    return objectness * confidence


def DecodeBBox(pred, idx, stride, lvl_idx, grid_x, grid_y, input_w, input_h, img_w, img_h, an_idx,
               grid_ws, grid_hs, bbox):
    bias = [220, 125, 128, 222, 264, 266,
            35, 87, 102, 96, 60, 170,
            10, 15, 24, 36, 72, 42]
    num_bias = 3
    idx = idx + 0
    box = [0., 0., 0., 0.]
    pred_x = pred[idx * 8 + 0]
    pred_y = pred[idx * 8 + int(stride)]
    pred_w = pred[idx * 8 + 2 * int(stride)]
    pred_h = pred[idx * 8 + 3 * int(stride)]
    # box[0] = (grid_x + Sigmoid(pred_x)) * img_w / grid_ws[lvl_idx]
    # box[1] = (grid_y + Sigmoid(pred_y)) * img_h / grid_hs[lvl_idx]
    # box[2] = math.exp(pred_w) * bias[lvl_idx * num_bias * 2 + an_idx*2] * img_w / input_w
    # box[3] = math.exp(pred_h) * bias[lvl_idx * num_bias * 2 + an_idx*2+1] * img_h / input_h

    box[0] = (grid_x + Sigmoid(pred_x)) * img_w / grid_ws[lvl_idx]
    box[1] = (grid_y + Sigmoid(pred_y)) * img_h / grid_hs[lvl_idx]
    box[2] = math.exp(pred_w) * bias[lvl_idx * num_bias * 2 + an_idx * 2] * img_w / input_w
    box[3] = math.exp(pred_h) * bias[lvl_idx * num_bias * 2 + an_idx * 2 + 1] * img_h / input_h

    # bbox.append(box[0] - box[2] * 0.5)
    # bbox.append(box[1] - box[3] * 0.5)
    # bbox.append(box[0] + box[2] * 0.5)
    # bbox.append(box[1] + box[3] * 0.5)
    # print (list(map(int, bbox[-4:])))
    box2 = []
    box2.append(box[0] - box[2] * 0.5)
    box2.append(box[1] - box[3] * 0.5)
    box2.append(box[0] + box[2] * 0.5)
    box2.append(box[1] + box[3] * 0.5)
    return box2


def computeIoU(bbox, id1, id_keep):
    box1 = bbox[id1][:-1]  # * 5:id1 * 5 + 4]
    box2 = bbox[id_keep][:-1]  # [id_keep*5:id_keep*5+4]
    l1, t1, r1, b1 = box1
    l2, t2, r2, b2 = box2
    l = max(l1, l2)
    t = max(t1, t2)
    r = min(r1, r2)
    b = min(b1, b2)

    if r <= l or b <= t:
        return 0.
    area1 = (r1 - l1) * (b1 - t1)
    area2 = (r2 - l2) * (b2 - t2)
    area_inter = (r - l) * (b - t)
    area_iou = area_inter / (area1 + area2 - area_inter)
    return area_iou


def NMS(scoreIndex, bboxes):
    # overlap1D
    # computeIOU
    # print("NMS", scoreIndex)
    # print("NMS", bboxes)
    indices = []
    num_thr = 0.5
    for item in scoreIndex:
        idx = item[1]
        keep = True
        for k in range(len(indices)):
            if keep:
                keptIdx = indices[k]
                overlap = computeIoU(bboxes, idx, keptIdx)
                keep = overlap <= num_thr
            else:
                break
        if keep:
            indices.append(idx)
    # print ("indices:", len(indices), indices)
    return indices


if __name__ == "__main__":
    """
    0 猫
    1 狗
    2 婴儿
    """
    onnx_file = "CatDogBaby0412V2Main_quant.onnx"

    # predict by ONNX Runtime
    ort_sess = onnxruntime.InferenceSession(onnx_file)

    predict_images_onnx(ort_sess)
