# eoding:utf-8
import copy

import numpy as np
import onnxruntime
import cv2
import math


def Sigmoid(x):
    try:
        y = 1. / (1. + math.exp(-x))
        return y
    except:
        return 1e-5


def CalConfScore(pred, idx, stride, cls_id):
    # objectness = Sigmoid(pred[idx])
    # confidence = Sigmoid(pred[idx + (1+cls_id)*int(stride)])

    objectness1 = (pred[idx * 6 + 4])
    confidence1 = (pred[idx * 6 + 4 + (1 + cls_id) * int(stride)])
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
    pred_x = pred[idx * 6 + 0]
    pred_y = pred[idx * 6 + int(stride)]
    pred_w = pred[idx * 6 + 2 * int(stride)]
    pred_h = pred[idx * 6 + 3 * int(stride)]
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


def predict_image_onnx(ort_sess, image):
    """
    :param net:
    :param path:
    :return:
    """
    img_w = image.shape[1]
    img_h = image.shape[0]
    image_disp = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_origin = cv2.resize(image, (320, 224))
    image = np.zeros(shape=[1, 1, 224, 320], dtype=np.float32)
    image[0][0] = image_origin

    ort_inputs = {ort_sess.get_inputs()[0].name: image}
    head_output = ort_sess.run(None, ort_inputs)

    data0 = head_output[0]
    data1 = head_output[1]
    data2 = head_output[2]

    num_cls = 1
    num_lvls = 3
    grid_hs = [7, 14, 28]
    grid_ws = [10, 20, 40]
    grid_ss = [1., 1., 1.]
    num_bias = 3
    conf_thr_table = [0.2, 0.2, 0.2]  # conf_thr_table = [0.005, 0.005, 0.005]
    det_w = 320
    det_h = 224
    box_rslt = []
    for c in range(num_cls):
        bboxes = []
        scoreIdx = []  # [[float, int]]
        bboxIdx = 0
        for i in range(num_lvls):
            gridH = grid_hs[i]
            gridW = grid_ws[i]
            fpStride = grid_ss[i]

            anStride = (num_cls + 5) * fpStride
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

    boxes = copy.deepcopy(box_rslt)
    box_dst = []
    for i, pred in enumerate(boxes):
        if pred[1] >= 0.32:
            l, t, r, b = boxes[i][2:]
            l, t, r, b = l * img_w, t * img_h, r * img_w, b * img_h
            box_dst.append([pred[1], l, t, r, b])
            l, t, r, b = map(int, [l, t, r, b])
            cv2.rectangle(image_disp, (l, t), (r, b), (0, 255, 255), 2, 8, 0)
            fscore = pred[1]
            fscore = round(fscore, 3)
    return box_dst

            # cv2.putText(image_disp, str(fscore) + "|" + str(r - l) + "|" + str(b - t), (l, t), 1, 3, (0, 0, 255), 3, 8,
            #             0)
    # print("boxdst:", box_dst)
    # cv2.imshow('preds', image_disp)
    # cv2.waitKey(0)


if __name__ == "__main__":
    onnx_file = "FaceDetection1107V4Main_noquant.onnx"
    # predict by ONNX Runtime
    ort_sess = onnxruntime.InferenceSession(onnx_file)

    predict_image_onnx(ort_sess)
    # print("Exported model has been predicted by ONNXRuntime!")
