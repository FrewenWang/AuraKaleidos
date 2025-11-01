"""
人脸检测-人脸点检测，顺序执行
"""

import copy
import math

import cv2
import onnxruntime
import cv2 as cv
import numpy as np


def get_facept_onnx(facept_Model, img_face, inputW, inputH):
    norm_w, norm_h = inputW, inputH
    pt_rslt = []
    data_draw = copy.deepcopy(img_face)
    data_out = cv.resize(img_face, (norm_w, norm_h))
    data = cv.cvtColor(data_out, cv.COLOR_BGR2GRAY)

    data2 = np.array(data, dtype=np.float32)
    inputdata = np.zeros(shape=[1, 1, norm_h, norm_w], dtype=np.float32)
    inputdata[0][0] = data2

    # inputdata.tofile("demo_face.raw")
    # inputdata = np.fromfile("demo_face.raw", dtype=np.float32)
    # inputdata = np.reshape(inputdata, (1, 1, norm_h, norm_w))

    ort_inputs = {facept_Model.get_inputs()[0].name: inputdata}
    output_rslt = facept_Model.run(None, ort_inputs)
    output_rslt1 = list(output_rslt[0][0])

    # output_rslt[0].tofile("demo_face_rslt.raw")
    # output_rslt_1 = np.fromfile("demo_face_rslt.raw", dtype=np.float32)
    # output_rslt1 = list(output_rslt_1)

    face_cls = output_rslt1[0]
    pitch = output_rslt1[-4] * 180 / math.pi
    yaw = output_rslt1[-3] * 180 / math.pi
    roll = output_rslt1[-2] * 180 / math.pi
    eye_status = output_rslt1[-1]
    # print ("人脸分数：", output_rslt1[0])
    # print("pitch-yaw-roll:", round(pitch, 2), round(yaw, 2), round(roll, 2))
    # print ("睁闭眼分数：", output_rslt1[-1])
    data_out_h, data_out_w, _ = data_draw.shape
    fratio_w = data_out_w * 1. / norm_w
    fratio_h = data_out_h * 1. / norm_h
    for i in range(106):
        px = output_rslt1[2 * i + 1]
        py = output_rslt1[2 * i + 2]
        pt_rslt.append([px * fratio_w, py * fratio_h])

        # px = int(px * data_out_w)
        # py = int(py * data_out_h)
        # cv.circle(data_draw, (int(px * fratio_w), int(py * fratio_h)), 3, (0, 0, 255), -1, 8, 0)
    # cv.imshow("faceImg_draw", data_draw)
    # cv.imwrite("/home/baidu/Downloads/不同的训练模型文件/人脸关键点/除255/加新数据/demo信息说明/" + str(id) + ".jpg", data_draw)
    # cv.waitKey(0)
    return [face_cls, pt_rslt, pitch, yaw, roll, eye_status]


class FaceDetectOnnx:
    def __init__(self, pathModel):
        self.ort_sess = onnxruntime.InferenceSession(pathModel)

    def Sigmoid(self, x):
        try:
            y = 1. / (1. + math.exp(-x))
            return y
        except:
            return 1e-5

    def CalConfScore(self, pred, idx, stride, cls_id):
        # objectness = Sigmoid(pred[idx])
        # confidence = Sigmoid(pred[idx + (1+cls_id)*int(stride)])

        objectness1 = (pred[idx * 6 + 4])
        confidence1 = (pred[idx * 6 + 4 + (1 + cls_id) * int(stride)])
        objectness = self.Sigmoid(objectness1)
        confidence = self.Sigmoid(confidence1)
        return objectness * confidence

    def DecodeBBox(self, pred, idx, stride, lvl_idx, grid_x, grid_y, input_w, input_h, img_w, img_h, an_idx,
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

        box[0] = (grid_x + self.Sigmoid(pred_x)) * img_w / grid_ws[lvl_idx]
        box[1] = (grid_y + self.Sigmoid(pred_y)) * img_h / grid_hs[lvl_idx]
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

    def computeIoU(self, bbox, id1, id_keep):
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

    def NMS(self, scoreIndex, bboxes):
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
                    overlap = self.computeIoU(bboxes, idx, keptIdx)
                    keep = overlap <= num_thr
                else:
                    break
            if keep:
                indices.append(idx)
        # print ("indices:", len(indices), indices)
        return indices

    def getfacebox(self, image):
        img_w = image.shape[1]
        img_h = image.shape[0]
        # print('original image shape:', image.shape)
        image_disp = image.copy()
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image_origin = cv.resize(image, (288, 160))
        image_pre = image_origin.astype(np.float32)
        # print('resized image shape:', image_pre.shape)
        image = np.zeros(shape=[1, 1, 160, 288], dtype=np.float32)
        image[0][0] = image_pre

        # im_shape = np.array([[160, 320]], dtype=np.float32)
        # scale_y = 160. / image_disp.shape[0]
        # scale_x = 320. / image_disp.shape[1]
        # scale_factor = np.array([[scale_y, scale_x]], dtype=np.float32)

        # ort_inputs = {ort_sess.get_inputs()[0].name: im_shape,
        #               ort_sess.get_inputs()[1].name: image,
        #               ort_sess.get_inputs()[2].name: scale_factor
        #               }

        ort_inputs = {self.ort_sess.get_inputs()[0].name: image}

        # boxes, num_boxes = ort_sess.run(None, ort_inputs)
        head_output = self.ort_sess.run(None, ort_inputs)

        num_cls = 1
        num_lvls = 3
        grid_hs = [5, 10, 20]
        grid_ws = [9, 18, 36]
        grid_ss = [1., 1., 1.]
        num_bias = 3
        conf_thr_table = [0.2, 0.2, 0.2]  # conf_thr_table = [0.005, 0.005, 0.005]
        det_w = 288
        det_h = 160
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
                            score = self.CalConfScore(outData, obj_idx, fpStride, c)
                            if score >= conf_thr_table[c]:
                                boxtmp = self.DecodeBBox(outData, obj_idx, fpStride, i,
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
            indices = self.NMS(scoreIdx, bboxes)
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

        boxes = copy.deepcopy(box_rslt)
        # print (boxes)
        boxes_rslt = []
        for i, pred in enumerate(boxes):
            if pred[1] >= 0.2:
                box_item_tmp = []
                l, t, r, b = boxes[i][2:]
                # centx = (l + r) / 2.
                # centy = (t + b) / 2.
                # h_pre = b-t
                # w_pre  = r-l
                # w_pre = 0.58*w_pre
                # l = max(0,  centx - w_pre/2.)
                # r = min(centx + w_pre/2., image_disp.shape[1])
                l, t, r, b = l * img_w, t * img_h, r * img_w, b * img_h
                box_item_tmp.append(pred[1])
                box_item_tmp.append(l)
                box_item_tmp.append(t)
                box_item_tmp.append(r)
                box_item_tmp.append(b)
                boxes_rslt.append(box_item_tmp)
                # l,t,r,b = map(int, [l,t,r,b])
                # cv.rectangle(image_disp, (l,t), (r,b), (0,255,255), 2, 8, 0)
                # fscore = pred[1]
                # fscore=round(fscore, 3)
                # cv.putText(image_disp, str(fscore)+"|"+str(r-l)+"|"+str(b-t), (l,t), 1, 3, (0,0,255), 3, 8, 0)

        # cv.imshow('preds', image_disp)
        # cv.waitKey(1)
        return boxes_rslt


def main_facept_img(ort_sess, img, boxitem):
    net_inputH = 192
    net_inputW = 192

    imgH, imgW, _ = img.shape
    # for boxitem in boxRslt:
    score, l, t, r, b = boxitem

    edge = 1.2 * max(r - l, b - t)
    centx = (l + r) / 2.
    centy = (t + b) / 2.
    l = centx - edge / 2.
    r = centx + edge / 2.
    t = centy - edge / 2.
    b = centy + edge / 2.
    l = min(max(0, l), imgW)
    t = min(max(0, t), imgH)
    r = min(max(0, r), imgW)
    b = min(max(0, b), imgH)

    l, t, r, b = map(int, [l, t, r, b])
    imgface = img[t:b, l:r, :]
    pt_rslt_ = get_facept_onnx(ort_sess, imgface, net_inputW, net_inputH)

    facecls = pt_rslt_[0]
    pt_rslt = pt_rslt_[1]
    pitch = pt_rslt_[2]
    yaw = pt_rslt_[3]
    roll = pt_rslt_[4]
    eye_status = pt_rslt_[5]
    points = []
    for id in range(len(pt_rslt)):
        px, py = pt_rslt[id]
        px, py = int(px + l), int(py + t)
        points.append((px, py))

            # cv.circle(img_draw, (px, py), 2, (0, 255, 0), -1, 8, 0)
        #     # 只画出偶数索引
        #     if id % 2 == 0:
        #         cv.putText(img_draw, str(id + 1), (px, py), 1, 1, (0, 0, 255), 1, 8, 0)
        # cv.putText(img_draw, "pitch:" + str(round(pitch, 3)), (100, 50), 1, 2, (0, 0, 255), 2, 8, 0)
        # cv.putText(img_draw, "yaw:" + str(round(yaw, 3)), (100, 90), 1, 2, (0, 0, 255), 2, 8, 0)
        # cv.putText(img_draw, "roll:" + str(round(roll, 3)), (100, 130), 1, 2, (0, 0, 255), 2, 8, 0)
        # cv.putText(img_draw, "facecls:" + str(round(facecls, 3)), (100, 170), 1, 2, (0, 0, 255), 2, 8, 0)  # 建议分值取>=0.1
        # cv.putText(img_draw, "eye:" + str(round(eye_status, 3)), (100, 210), 1, 2, (0, 0, 255), 2, 8, 0)  # 睁闭眼判别阈值，建议取
        # cv.rectangle(img_draw, (l, t), (r, b), (0, 255, 255), 3, 8, 0)
        # fscore = round(score, 3)
        # cv.putText(img_draw, str(fscore) + "|" + str(r - l) + "|" + str(b - t), (l, t), 1, 3, (0, 0, 255), 3, 8, 0)
    # cv.imshow("img_draw", img_draw)
    # cv.waitKey(0)
    return points, {"pitch": pitch, "yaw": yaw, "roll": roll}

if __name__ == "__main__":
    main_facept_img()
