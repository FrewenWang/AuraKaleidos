"""
人脸检测-人脸点检测-EyeCenter，顺序执行,
"""

import copy
import math
import onnxruntime
import cv2 as cv
import numpy as np
import os
import glob
import json

from predict_images_emo import Emotion
emo = Emotion()
# from predict_images_onnx_0307_quality import DMSDanger
# quality = DMSDanger()


def judging_camera_block(img, cam_type):
    """
    Main funtion of algorithm
    :param img: input image
    :param cam_type: 'oms_ir', 'oms_rgb', 'dms'.
    :return: if_block, block_score, block_ratio
    """
    # Trans to gray scale and resize
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # For RGB image: 3 channels
    else:
        img_gray = img  # For IR image: single channel
    target_size = (100, 100)
    img_gray = cv.resize(img_gray, dsize=target_size, interpolation=cv.INTER_AREA)

    # Judging algorithms
    img_gray_np = np.array(img_gray, dtype=float)
    img_gaussblur_np = cv.GaussianBlur(img_gray_np, ksize=(3, 3), sigmaX=1000.0)
    delta_img = abs(img_gray_np - img_gaussblur_np)

    block_score = np.mean(delta_img)
    is_block = False
    block_ratio = 0  # used in dms camera
    block_score_list = []  # block score for multi zones
    if cam_type == 'oms_ir':
        grid_num = 15
        thresh_hold_multizone = 1.0
        thresh_grid_ratio = 0.56
        block_num = 0
        grid_width = int(target_size[0] / grid_num)
        grid_height = int(target_size[1] / grid_num)
        start_row = 4
        for i in range(start_row, grid_num):
            interval_h_start = i * grid_height
            interval_h_end = (i + 1) * grid_height
            for j in range(grid_num):
                interval_w_start = j * grid_width
                interval_w_end = (j + 1) * grid_width
                delta_block = np.mean(delta_img[interval_h_start: interval_h_end, interval_w_start: interval_w_end])
                if delta_block < thresh_hold_multizone:
                    block_num += 1
                block_score_list.append(delta_block)
        block_ratio = float(block_num / ((grid_num - start_row) * grid_num))  # ratio of blocked zones
        if block_ratio >= thresh_grid_ratio:
            is_block = True  # The camera is blocked
    elif cam_type == 'oms_rgb':
        grid_num = 15
        thresh_hold_multizone = 1.5
        thresh_grid_ratio = 0.25
        block_num = 0
        grid_width = int(target_size[0] / grid_num)
        grid_height = int(target_size[1] / grid_num)
        start_row = 4
        for i in range(start_row, grid_num):
            interval_h_start = i * grid_height
            interval_h_end = (i + 1) * grid_height
            for j in range(grid_num):
                interval_w_start = j * grid_width
                interval_w_end = (j + 1) * grid_width
                delta_block = np.mean(delta_img[interval_h_start: interval_h_end, interval_w_start: interval_w_end])
                if delta_block < thresh_hold_multizone:
                    block_num += 1
                block_score_list.append(delta_block)
        block_ratio = float(block_num / ((grid_num - start_row) * grid_num))  # ratio of blocked zones
        if block_ratio >= thresh_grid_ratio:
            is_block = True  # The camera is blocked
    elif cam_type == 'dms':
        grid_num = 15
        thresh_hold_multizone = 1.5
        thresh_grid_ratio = 0.71
        block_num = 0
        grid_width = int(target_size[0] / grid_num)
        grid_height = int(target_size[1] / grid_num)
        center_block_num = 0
        center_inds_up = 6
        center_inds_left = 4
        for i in range(grid_num):
            interval_h_start = i * grid_height
            interval_h_end = (i + 1) * grid_height
            for j in range(grid_num):
                interval_w_start = j * grid_width
                interval_w_end = (j + 1) * grid_width
                delta_block = np.mean(delta_img[interval_h_start: interval_h_end, interval_w_start: interval_w_end])
                if delta_block < thresh_hold_multizone:
                    block_num += 1
                block_score_list.append(delta_block)
                if i >= center_inds_up and j >= center_inds_left:
                    if delta_block < thresh_hold_multizone:
                        center_block_num += 1
        block_ratio = float(center_block_num / ((grid_num - center_inds_up) * (grid_num - center_inds_left)))

        # Judge whether block
        if block_ratio >= thresh_grid_ratio:
            is_block = True  # The camera is blocked
            up = np.mean(img_gray[int(0.45 * target_size[1]): int(0.60 * target_size[1]),
                         int(0.4 * target_size[0]): int(0.6 * target_size[0])])

            down = np.mean(img_gray[int(0.8 * target_size[1]): int(0.93 * target_size[1]),
                           int(0.4 * target_size[0]): int(0.6 * target_size[0])])
            if down > 100 and down / up > 10:  # avoid no driver recognized occluded
                is_block = False
        else:
            is_block = False  # The camera is unblocked
    return [is_block, block_score, block_ratio, block_score_list]


def get_facept_onnx(facept_Model, img_face, inputW, inputH):
    """get_facept_onnx"""
    norm_w, norm_h = inputW, inputH
    pt_rslt = []
    data_out = cv.resize(img_face, (norm_w, norm_h))
    data = cv.cvtColor(data_out, cv.COLOR_BGR2GRAY)
    data2 = data
    inputdata = np.zeros(shape=[1, 1, norm_h, norm_w], dtype=np.float32)
    inputdata[0][0] = data2

    # inputdata.tofile("demo_face1.raw")
    # inputdata = np.fromfile("demo_face1.raw", dtype=np.float32)
    # inputdata = np.reshape(inputdata, (1, 1, 128, 128))

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
    data_out_h, data_out_w, _ = img_face.shape
    fratio_w = data_out_w * 1. / norm_w
    fratio_h = data_out_h * 1. / norm_h
    for i in range(106):
        px = output_rslt1[2 * i + 1]
        py = output_rslt1[2 * i + 2]
        pt_rslt.append([px * fratio_w, py * fratio_h])
    return [face_cls, pt_rslt, pitch, yaw, roll, eye_status]


def get_pupil_onnx(ort_sess_pupil, img, facept_eye, OPEN_CALSS, is_left, fuse_bn_1layer):
    """get_pupil_onnx"""
    facept_eye = np.array(facept_eye)
    imgH, imgW, _ = img.shape
    l = np.min(facept_eye[:, 0])
    t = np.min(facept_eye[:, 1])
    r = np.max(facept_eye[:, 0])
    b = np.max(facept_eye[:, 1])
    centx = (l + r) / 2.
    centy = (t + b) / 2.
    edge = 1.4 * max(r - l, b - t)
    l = centx - edge / 2.
    t = centy - edge / 2.
    r = centx + edge / 2.
    b = centy + edge / 2.
    l = min(max(0, int(l)), imgW)
    t = min(max(0, int(t)), imgH)
    r = min(max(0, int(r)), imgW)
    b = min(max(0, int(b)), imgH)
    imgEye = img[t:b, l:r, :]
    h_ori, w_ori, _ = imgEye.shape
    if h_ori < 10 or w_ori < 10:
        print('eye crop image is too small!, h:%d, w:%d' % (h_ori, w_ori))
        return None

    norm_w, norm_h = 96, 96
    eyept_rslt = []
    data_draw = copy.deepcopy(imgEye)
    data_out = cv.resize(imgEye, (norm_w, norm_h))

    data = cv.cvtColor(data_out, cv.COLOR_BGR2GRAY)

    if fuse_bn_1layer:
        data2 = np.array(data, dtype=np.float32) / 1.0
    else:
        data2 = np.array(data, dtype=np.float32) / 255.

    inputdata = np.zeros(shape=[1, 1, norm_h, norm_w], dtype=np.float32)
    inputdata[0][0] = data2

    # if is_left:
    #     cv.imwrite('/home/baidu/Desktop/chengq/data/eyeCenter_model/0722/demo_leftEye.jpg', data2)
    #     inputdata.tofile("/home/baidu/Desktop/chengq/data/eyeCenter_model/0722/demo_leftEye.raw")
    # if not is_left:
    #     cv.imwrite('/home/baidu/Desktop/chengq/data/eyeCenter_model/0722/demo_rightEye.jpg', data2)
    #     inputdata.tofile("/home/baidu/Desktop/chengq/data/eyeCenter_model/0722/demo_rightEye.raw")

    # inputdata = np.fromfile("demo_rightEye.raw", dtype=np.float32)
    # inputdata = np.reshape(inputdata, (1, 1, 96, 96))

    ort_inputs = {ort_sess_pupil.get_inputs()[0].name: inputdata}
    output_rslt = ort_sess_pupil.run(None, ort_inputs)
    output_rslt1 = list(output_rslt[0][0])

    # output_rslt[0].tofile("rslt_rightEye.raw")
    # output_rslt_1 = np.fromfile("rslt_rightEye.raw", dtype=np.float32)
    # output_rslt1 = list(output_rslt_1)

    if OPEN_CALSS == 2:
        pupil_cls = output_rslt1[0]
        eye_pts = output_rslt1[1:19]
        eye_status = output_rslt1[19]
    else:
        eye_pts = output_rslt1[3:21]
        pred_pupil_class = output_rslt1[:3]
        prob_pupil = np.exp(pred_pupil_class) / np.exp(pred_pupil_class).sum()  # 归一化
        pupil_cls = prob_pupil.argsort()[-1].tolist()
        pred_eye_class = output_rslt1[21:]
        prob_eye = np.exp(pred_eye_class) / np.exp(pred_eye_class).sum()  # 归一化
        eye_status = prob_eye.argsort()[-1].tolist()

    data_out_h, data_out_w, _ = data_draw.shape
    for i in range(9):
        px = eye_pts[2 * i]
        py = eye_pts[2 * i + 1]
        # eyept_rslt.append([px * data_out_w + l, py * data_out_h + t])
        eyept_rslt.append([px * data_out_w / norm_w + l, py * data_out_h / norm_w + t])

        if draw_image:
            if i == 0:
                cv.circle(data_draw, (int(px * data_out_w / norm_w), int(py * data_out_h / norm_w)), 3, (0, 0, 255), -1, 8, 0)
            else:
                cv.circle(data_draw, (int(px * data_out_w / norm_w), int(py * data_out_h / norm_w)), 2, (0, 255, 0), -1, 8, 0)
                cv.putText(data_draw, "%d" % i, (int(px * data_out_w/norm_w), int(py * data_out_h/norm_w)), 1, 0.8, (0, 0, 255), 1, 8, 0)
    
    return [pupil_cls, eyept_rslt, eye_status]


class FaceDetectOnnx:
    """FaceDetectOnnx"""
    def __init__(self, pathModel):
        self.ort_sess = onnxruntime.InferenceSession(pathModel)

    def Sigmoid(self, x):
        """Sigmoid"""
        try:
            y = 1. / (1. + math.exp(-x))
            return y
        except:
            return 1e-5

    def CalConfScore(self, pred, idx, stride, cls_id):
        """CalConfScore"""
        # objectness = Sigmoid(pred[idx])
        # confidence = Sigmoid(pred[idx + (1+cls_id)*int(stride)])

        objectness1 = (pred[idx * 6 + 4])
        confidence1 = (pred[idx * 6 + 4 + (1 + cls_id) * int(stride)])
        objectness = self.Sigmoid(objectness1)
        confidence = self.Sigmoid(confidence1)
        return objectness * confidence

    def DecodeBBox(self, pred, idx, stride, lvl_idx, grid_x, grid_y, input_w, input_h, img_w, img_h, an_idx,
                   grid_ws, grid_hs, bbox):
        """DecodeBBox"""
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
        """computeIoU"""
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
        """NMS"""
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
        """getfacebox"""
        img_w = image.shape[1]
        img_h = image.shape[0]
        # print('original image shape:', image.shape)
        image_disp = image.copy()
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image_origin = cv.resize(image, (320, 224))
        image_pre = image_origin.astype(np.float32)
        # print('resized image shape:', image_pre.shape)
        image = np.zeros(shape=[1, 1, 224, 320], dtype=np.float32)
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
            if pred[1] >= 0.33:
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


def deal_single_image(img, path, OPEN_CALSS, fuse_bn_1layer, res_save_path):
    """deal_single_image"""
    global total, count, count_left, count_right, check_count, check_fps, video_fps, is_oms, is_oms_left
    pred_res = {}
    check_count += 1
    img_drawpt = copy.deepcopy(img)
    imgH, imgW, _ = img.shape
    if is_oms:  # todo
        # cover half
        row = img.shape[0]
        col = img.shape[1]
        mask = np.zeros((row, int(col / 2), 3), dtype=np.int8)
        img[:, int(col / 2):, :] = mask  # cover you

    boxRslt = facedetect.getfacebox(image=img)
    boxid = -1
    box_width = 300
    eye_close = 0
    for boxitem in boxRslt:
        boxid += 1
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

        pt_rslt_ = get_facept_onnx(ort_sess_facept, imgface, net_inputW, net_inputH)
        facecls = pt_rslt_[0]
        pt_rslt = pt_rslt_[1]
        pitch = pt_rslt_[2]
        yaw = pt_rslt_[3]
        roll = pt_rslt_[4]
        eye_status = pt_rslt_[5]

        facepts_image = []
        for id in range(len(pt_rslt)):
            px, py = pt_rslt[id]
            facepts_image.append([px + l, py + t])
            px, py = int(px + l), int(py + t)
            if draw_image:
                cv.circle(img_drawpt, (px, py), 2, (0, 255, 0), -1, 8, 0)
                # 只画出偶数索引
                if id % 2 == 0:
                    cv.putText(img_drawpt, str(id + 1), (px, py), 1, 1, (0, 0, 255), 1, 8, 0)

        # emo ------------------------------------
        emo_class, emo_prob = emo.predict_camera_emo(EmoNet, facepts_image, img) #Emo_labeled = {0: 'other', 1: 'joy', 2: 'surprise', 3: 'sad', 4: 'angry'}
        if draw_image:
            cv.putText(img_drawpt, "emo:" + str(emo_class), (30, 750), 1, 2, (0, 0, 255), 2, 8, 0)
            cv.putText(img_drawpt, "emo_prob:" + str(round(emo_prob, 3)), (300, 750), 1, 2, (0, 0, 255), 2, 8, 0)
        
        # 后处理修正，有感活检使用原始输出，不进行后处理校正
        #############################################33
        if pitch > 0:
            pitch += 5
        if yaw > 0:
            yaw += 7.5
        else:
            yaw -= 7.5
        ##########################################
        # roll值计算更改为：左眼的点与右眼的点的直线的角度，不再使用模型直接输出的值
        px1 = []
        py1 = []
        for i in range(51, 59):
            px1.append(pt_rslt[i][0])
            py1.append(pt_rslt[i][1])
        px2 = []
        py2 = []
        for i in range(61, 69):
            px2.append(pt_rslt[i][0])
            py2.append(pt_rslt[i][1])
        px1 = sum(px1) / (len(px1) + 1e-6)
        py1 = sum(py1) / (len(py1) + 1e-6)
        px2 = sum(px2) / (len(px2) + 1e-6)
        py2 = sum(py2) / (len(py2) + 1e-6)
        if px1 == px2:
            px1 = px1 + 1e-6
        roll = math.atan((py1 - py2) / (px1 - px2)) * 180 / math.pi
        #################################################################
        # filter back person
        landmars_array = np.array(pt_rslt)
        landmark_rect = [min(landmars_array[:, 0]), min(landmars_array[:, 1]),
                         max(landmars_array[:, 0]), max(landmars_array[:, 1])]
        face_height = landmark_rect[3] - landmark_rect[1]
        if face_height < 110:
            if not (face_height > 95 and pitch < -10):  # pitch, up angle is negative.
                continue

        if draw_image:
            cv.putText(img_drawpt, "pitch:" + str(round(pitch, 3)), (100 + boxid * box_width, 50), 1, 2,
                       (0, 0, 255), 2, 8, 0)
            cv.putText(img_drawpt, "yaw:" + str(round(yaw, 3)), (100 + boxid * box_width, 90), 1, 2, (0, 0, 255), 2,
                       8, 0)
            cv.putText(img_drawpt, "roll:" + str(round(roll, 3)), (100 + boxid * box_width, 130), 1, 2, (0, 0, 255),
                       2, 8, 0)
            cv.putText(img_drawpt, "facecls:" + str(round(facecls, 3)), (100 + boxid * box_width, 170), 1, 2,
                       (0, 0, 255), 2, 8,
                       0)  # 建议分值取>=0.1
            cv.putText(img_drawpt, "eye:" + str(round(eye_status, 3)), (100 + boxid * box_width, 210), 1, 2,
                       (0, 0, 255), 2, 8,
                       0)  # 睁闭眼判别阈值，建议取
            cv.rectangle(img_drawpt, (l, t), (r, b), (0, 255, 255), 3, 8, 0)
            fscore = round(score, 3)
            cv.putText(img_drawpt, str(fscore) + "|" + str(r - l) + "|" + str(b - t), (l, t), 1, 3, (0, 0, 255), 3,
                       8, 0)

        # 分析瞳孔点
        # for i in range(106):
        #     px,py = facepts_image[i]
        #     px,py = int(px), int(py)
        #     cv.circle(img_drawpupil, (px,py), 3, (0,0,255), -1, 8, 0)

        facept_lefteye = facepts_image[51:59]
        facept_righteye = facepts_image[61:69]
        face_length = face_height  # use height

        # 左瞳孔结果，第一维度是否存在瞳孔，第2,3个值是瞳孔点，第4-19个值是眼睛周围点，第20个值是判断有无眼睛
        # 如果第20个值大于0.5则认为是眼睛，如果是眼睛则判别第一个值，如果第一个值大于0.4则认为存在瞳孔
        rslt_leftpupil = get_pupil_onnx(ort_sess_pupil, img, facept_lefteye, OPEN_CALSS, is_left=1, fuse_bn_1layer=fuse_bn_1layer)
        # print("left:", rslt_leftpupil)
        if rslt_leftpupil is not None:
            # use EyeCenter model, 9 points, eyecenter + 8 round points. the last two points are up and down center.
            left_eye_open_dist = np.sqrt((rslt_leftpupil[1][8][0] - rslt_leftpupil[1][7][0]) ** 2 +
                                         (rslt_leftpupil[1][8][1] - rslt_leftpupil[1][7][1]) ** 2)
            left_eye_open_norm = left_eye_open_dist / face_length * 500  # normalized to 500 pixels

            # use face landmark model
            left_eye_open_dist_f = np.sqrt((facepts_image[58 - 1][0] - facepts_image[59 - 1][0]) ** 2 +
                                       (facepts_image[58 - 1][1] - facepts_image[59 - 1][1]) ** 2)
            left_eye_open_norm_f = left_eye_open_dist_f / face_length * 500

            if draw_image:
                cv.putText(img_drawpt, 'landmark: w: %.0f, h: %.0f' % (landmark_rect[2] - landmark_rect[0],
                                                                       landmark_rect[3] - landmark_rect[1]),
                           (l, b + 40), 1, 2, (0, 0, 255), 2, 8, 0)
                cv.putText(img_drawpt, 'landmark: left_eye:' + " dist_origin:" + str(round(left_eye_open_dist_f, 3))
                           + ", dist_norm:" + str(round(left_eye_open_norm_f, 3)),
                           (l, b + 80), 1, 2, (0, 0, 255), 2, 8, 0)
        else:
            rslt_leftpupil = [0, 0, 0]  # not eye
            left_eye_open_dist = left_eye_open_norm = left_eye_open_dist_f = left_eye_open_norm_f = 0

        # cv.waitKey(200)
        # exit()
        # # 右瞳孔结果，结果同左瞳孔结果
        rslt_rightpupil = get_pupil_onnx(ort_sess_pupil, img, facept_righteye, OPEN_CALSS, is_left=0, fuse_bn_1layer=fuse_bn_1layer)
        # print("right:", rslt_rightpupil)
        if rslt_rightpupil is not None:
            # use EyeCenter model, 9 points, eyecenter + 8 round points. the last two points are up and down center.
            right_eye_open_dist = np.sqrt((rslt_rightpupil[1][8][0] - rslt_rightpupil[1][7][0]) ** 2 +
                                          (rslt_rightpupil[1][8][1] - rslt_rightpupil[1][7][1]) ** 2)
            right_eye_open_norm = right_eye_open_dist / face_length * 500

            right_eye_open_dist_f = np.sqrt((facepts_image[68 - 1][0] - facepts_image[69 - 1][0]) ** 2 +
                                        (facepts_image[68 - 1][1] - facepts_image[69 - 1][1]) ** 2)
            right_eye_open_norm_f = right_eye_open_dist_f / face_length * 500

            if draw_image:
                cv.putText(img_drawpt, 'landmark: w: %.0f, h: %.0f' % (landmark_rect[2] - landmark_rect[0],
                                                                       landmark_rect[3] - landmark_rect[1]),
                           (l, b + 40), 1, 2, (0, 0, 255), 2, 8, 0)
                cv.putText(img_drawpt, 'landmark: right_eye:' + " dist_origin:" + str(round(right_eye_open_dist_f, 3))
                           + ", dist_norm:" + str(round(right_eye_open_norm_f, 3)),
                           (l, b + 120), 1, 2, (0, 0, 255), 2, 8, 0)
        else:
            rslt_rightpupil = [0, 0, 0]  # not eye
            right_eye_open_dist = right_eye_open_norm = right_eye_open_dist_f = right_eye_open_norm_f = 0

        # meixin to DMS line
        y_pos = int(1300 * (1 - 0.23))  # DMS 21.5%       
        dist_l_eye = y_pos - facepts_image[60 - 1][1]
        dist_r_eye = y_pos - facepts_image[70 - 1][1]
        meixin_pos = y_pos - facepts_image[72 - 1][1]
        mouth_open_dist = np.sqrt((facepts_image[100 - 1][0] - facepts_image[103 - 1][0]) ** 2 +
                                  (facepts_image[100 - 1][1] - facepts_image[103 - 1][1]) ** 2)
        mouth_open_norm_dist = mouth_open_dist / face_length * 500
        if draw_image:# and not is_oms:
            cv.line(img, (0, y_pos), (1600, y_pos), color=(0, 255, 0))
            cv.line(img_drawpt, (0, y_pos), (1600, y_pos), color=(0, 255, 0))
            cv.putText(img_drawpt, "dist: l: %.2f, m: %.2f, r: %.2f, mouth_open: %.2f" % 
                       (dist_l_eye, meixin_pos, dist_r_eye, mouth_open_norm_dist),
                       (100 + boxid * box_width, 400), 1, 2, (0, 0, 255), 2, 8, 0)

        # close eye methods
        pred_res = {'ec_l_eye': rslt_leftpupil[2], 'ec_l_pupil': rslt_leftpupil[0], 'ec_l_dist_norm': left_eye_open_norm,
                    'ec_r_eye': rslt_rightpupil[2], 'ec_r_pupil': rslt_rightpupil[0], 'ec_r_dist_norm': right_eye_open_norm,
                    'facept_eye_status': eye_status,
                    'facept_l_dist_norm': left_eye_open_norm_f, 'facept_r_dist_norm': right_eye_open_norm_f,
                    'yaw': yaw, 'pitch': pitch, 'roll': roll,
                    'meixin_pos': meixin_pos, 'mouth_open': mouth_open_norm_dist,
                    'emo_class': emo_class, 'emo_prob': emo_prob
                    }
        
        # 0715：通过高斯差分判断图像遮挡
        cam_type = "oms_rgb" if is_oms else "dms"
        is_block, block_score, block_ratio, _ = judging_camera_block(img, cam_type)
        if is_oms:  # OMS情况下，在预处理阶段就会将右边图像mask掉
            pred_res["occlusion"] = False
        else:
            pred_res["occlusion"] = True if is_block else False

        eye_close, left_close, right_close = is_close_eye(pred_res,
                                                          thresh_if_eye=0.5,
                                                          OPEN_CALSS=OPEN_CALSS,
                                                          emo_class=emo_class,
                                                          emo_prob=emo_prob
                                                          )
        pred_res['eye_close'] = eye_close
        
        if draw_image:
            colors = [(0, 0, 255), (0, 255, 0)]
            cv.putText(img_drawpt,
                       "l: if_close:" + str(round(rslt_leftpupil[0], 3)) + ", if_eye:" + str(
                           round(rslt_leftpupil[2], 3))
                       + ", dist_origin:" + str(round(left_eye_open_dist, 3))
                       + ", dist_norm:" + str(round(left_eye_open_norm, 3)),
                       (100 + boxid * box_width, 260), 1, 2, colors[left_close], 2, 8, 0)
            cv.putText(img_drawpt,
                       "r: if_close:" + str(round(rslt_rightpupil[0], 3)) + ", if_eye:" + str(
                           round(rslt_rightpupil[2], 3))
                       + ", dist_origin:" + str(round(right_eye_open_dist, 3))
                       + ", dist_norm:" + str(round(right_eye_open_norm, 3)),
                       (100 + boxid * box_width, 310), 1, 2, colors[right_close], 2, 8, 0)

        if eye_close:
            count += 1
        if left_close:
            count_left += 1
        if right_close:
            count_right += 1

        close_eye_window.append(eye_close)
        if draw_image:
            cv.putText(img_drawpt,
                       "eye_close: %d" % eye_close,
                       (100 + boxid * box_width, 350), 1, 2, colors[eye_close], 2, 8, 0)
        if save_result:
            if eye_close:
                dst = res_save_path + '/results_' + path.split('/')[-2]
                if not os.path.exists(dst):
                    os.makedirs(dst)
                cv.imwrite(dst + '/' + path.split('/')[-1], img_drawpt)

    return pred_res


def is_close_eye(pred_dict, thresh_if_eye=0.5, OPEN_CALSS=3, emo_class=None, emo_prob=None):
    """close eye judge methods"""
    ec_l_eye = float(pred_dict['ec_l_eye'])
    ec_l_pupil = float(pred_dict['ec_l_pupil'])
    ec_l_dist_norm = float(pred_dict['ec_l_dist_norm'])
    ec_r_eye = float(pred_dict['ec_r_eye'])
    ec_r_pupil = float(pred_dict['ec_r_pupil'])
    ec_r_dist_norm = float(pred_dict['ec_r_dist_norm'])
    occlusion = bool(pred_dict["occlusion"])
    pitch = float(pred_dict["pitch"])

    # facept_eye_status = float(pred_dict['facept_eye_status'])
    # facept_l_dist_norm = float(pred_dict['facept_l_dist_norm'])
    # facept_r_dist_norm = float(pred_dict['facept_r_dist_norm'])

    left_close = 0
    right_close = 0

    if is_oms:
        if OPEN_CALSS == 2:
            thresh_if_eye = 0.6
            thresh_if_pupil = 0.45

            if ec_l_eye > thresh_if_eye and ec_l_pupil < thresh_if_pupil:  #
                left_close = 1
            if ec_r_eye > thresh_if_eye and ec_r_pupil < thresh_if_pupil:  #
                right_close = 1

            mouth_open = float(pred_dict['mouth_open'])
            if mouth_open > 40:
                left_close = 0
                right_close = 0

            if float(pred_dict['yaw']) > 55:
                left_close = 0
                right_close = 0

    else:
        if OPEN_CALSS == 2:
            # -----------------231224------------------------------
            thresh_if_eye = 0.5
            eye_close_dist_thresh = 6.5
            if ec_l_eye > thresh_if_eye and ec_l_pupil < 0.4:  #
                left_close = 1
            if ec_r_eye > thresh_if_eye and ec_r_pupil < 0.4:  #
                right_close = 1

            # 231109, add avoid false report.
            meixin_pos = float(pred_dict['meixin_pos'])
            mouth_open = float(pred_dict['mouth_open'])
            if meixin_pos < 50 or mouth_open > 60:
                left_close = 0
                right_close = 0

            if ec_l_eye > thresh_if_eye and ec_l_pupil < 0.4 \
                    and ec_l_dist_norm < eye_close_dist_thresh:  #
                left_close = 1
            if ec_r_eye > thresh_if_eye and ec_r_pupil < 0.4 \
                    and ec_r_dist_norm < eye_close_dist_thresh:  #
                right_close = 1
                
    # print('emo_class, emo_prob, mouth_open', emo_class, emo_prob, mouth_open)
    if emo_class in ['joy', 'angry', 'surprise']:#, ,  # and emo_prob > 0.95: #, 'angry' , 'surprise'   and emo_prob > 0.5  and emo_prob > 0.9 ,
        mouth_open = float(pred_dict['mouth_open'])
        if mouth_open > 18: #15 28 18
            left_close = 0
            right_close = 0

    right_close_dan = 0
    left_close_dan = 0
    if ec_l_eye < thresh_if_eye and right_close:
        right_close_dan = 1
    if ec_r_eye < thresh_if_eye and left_close:
        left_close_dan = 1

    eye_close = 0
    if (left_close and right_close) or right_close_dan or left_close_dan:
        eye_close = 1
    
    # dms情况下，人脸区域发生遮挡，则不做闭眼判断
    if (not is_oms) and occlusion:
        eye_close = 0
        left_close = 0
        right_close = 0
    # dms情况下，抬头角度过大
    if pitch < -40:
        eye_close = 0
        left_close = 0
        right_close = 0
    return eye_close, left_close, right_close


def eval_by_json_QA(events_result_path, OPEN_CALSS=3):
    files = glob.glob(events_result_path + '/result*.txt')
    files.sort()
    # thresholds = np.linspace(3.0, 5.0, 21)
    # thresholds = np.linspace(0.5, 0.65, 16)
    thresholds = [0.5]

    global video_fps, check_fps
    print('is_oms', is_oms)

    # for thresh in thresholds:
    #     print(thresh)
    total = 0
    close_img, close2close, close2open = 0, 0, 0
    open_img, open2open, open2close = 0, 0, 0
    count_have_report_event = 0
    for event_result_file in files:
        pred_results = read_pred_file(event_result_file)
        close_eye_window_event = []
        report_frames = []
        have_report_event = 0
        check_count = 0
        frame_count = 0
        for pred in pred_results:
            key = pred['key']

            if len(pred) < 3:  # no face
                eye_close = 0
            else:
                emo_class = pred['emo_class']
                emo_prob = float(pred['emo_prob'])
                close_result = is_close_eye(pred,
                                            thresh_if_eye=0.5,
                                            OPEN_CALSS=OPEN_CALSS,
                                            emo_class=emo_class,
                                            emo_prob=emo_prob
                                            )
                eye_close = close_result[0]

            if eye_close == 1:
                have_report_event = 1
                report_close_eye = 1
                report_frames.append(key)

        if have_report_event:
            count_have_report_event += 1

        if 'biyan' in event_result_file:
            close_img += 1
            if have_report_event:
                close2close += 1
            else:
                close2open += 1
        else:
            open_img += 1
            if have_report_event:
                open2close += 1
            else:
                open2open += 1
        total += 1
        
    if is_oms:
        open2close = open2close - 96
        open_img = open_img - 96
    print('close_img, close2close, close2open, recall, precision', close_img, close2close, close2open, float(close2close)/float(close_img+0.000001), float(close2close)/float(close2close+open2close+0.000001))
    print('open_img, open2open, open2close, recall, precision', open_img, open2open, open2close, float(open2open)/float(open_img+0.000001), float(open2open)/float(open2open+close2open+0.000001))

    return


def eval_by_json_file(events_result_path, if_need_sample=1, OPEN_CALSS=3):
    """eval_by_json_file, inferenced json result file, count tp fp tn fn.
                          find thresholds """
    files = glob.glob(events_result_path + '/result*.mp4.txt')  # 10个case
    files.sort()
    # thresholds = np.linspace(3.0, 5.0, 21)
    # thresholds = np.linspace(0.5, 0.65, 16)
    # thresholds = np.linspace(0.5, 0.95, 10)
    thresholds = [0.5]
    global video_fps, check_fps

    for thresh in thresholds:
        print(thresh)
        total = 0
        count_have_report_event = 0
        for event_result_file in files:
            pred_results = read_pred_file(event_result_file)
            close_eye_window_event = []
            report_frames = []
            have_report_event = 0
            check_count = 0
            frame_count = 0
            for pred in pred_results:
                key = pred['key']
                # print('pred', pred)

                if if_need_sample:
                    # check [0, 3, 5, 8]
                    mod_ind = frame_count % 10
                    frame_count += 1
                    if mod_ind not in [0, 3, 5, 8]:
                        continue

                if len(pred) < 3:  # no face
                    eye_close = 0
                else:
                    emo_class = pred['emo_class']
                    emo_prob = float(pred['emo_prob'])
                    close_result = is_close_eye(pred,
                                                thresh_if_eye=0.5,
                                                OPEN_CALSS=OPEN_CALSS,
                                                emo_class=emo_class,
                                                emo_prob=emo_prob
                                                )
                    eye_close = close_result[0]

                close_eye_window_event.append(eye_close)
                report_close_eye = 0
                close_ratio = 0
                if len(close_eye_window_event) > window_length:
                    close_eye_window_event.pop(0)
                    close_ratio = 1.0 * sum(close_eye_window_event) / len(close_eye_window_event)
                    if close_ratio >= 0.8:
                        # if "00005" in event_result_file:
                        #     print(key, eye_close, close_ratio, close_eye_window_event, len(close_eye_window_event))
                        have_report_event = 1
                        report_close_eye = 1
                        report_frames.append(key)
            if have_report_event:
                count_have_report_event += 1
            total += 1
            if have_report_event:
                print(os.path.basename(event_result_file), 'pred_results:', len(pred_results), len(report_frames))
                # print('report time:', len(report_frames), 'report_frames:', report_frames)
        print(events_result_path)
        print('videos total: %d, close_eye_report: %d' % (total, count_have_report_event))
    return


def eval_jidu_testdata_by_json_file(events_result_path, if_need_sample=0, OPEN_CALSS=3):
    """eval_by_json_file, inferenced json result file, count tp fp tn fn.
                          find thresholds """
    ignore_list = [
"dump_1612847612_15_1.mp4",  # 闭眼 FP
"dump_4376520089_15_1.mp4",  # 低头闭眼 FP
"Video_2023-03-03_11-54-41_15_1.mp4",  # 低头闭眼 FP
"dump_20230411122433_15_1.mp4",  # close eye FP
"dump_20230413200001_15_1.mp4",  # 摄像头太暗 FP
"dump_2396420827_15_1.mp4",  # 低头闭眼 FP
"dump_3795545751_15_1.mp4",  # 低头闭眼 FP
"dump_20230320202455_15_1.mp4",  # 低头闭眼 FP
'dump_20230522203213_15_1.mp4',  # 摄像头太暗 FP
'Video_2023-03-01_21-08-26_15_1.mp4', # 摄像头太暗 FP
'Video_2023-03-01_21-11-09_15_1.mp4',  # 摄像头太暗 FP
'Video_2023-03-01_21-13-58_15_1.mp4',  # 摄像头太暗 FP
'Video_2023-03-01_20-45-47_15_1.mp4',  # 摄像头太暗 FP
'Video_2023-03-01_20-53-29_15_1.mp4',  # 摄像头太暗 FP
'Video_2023-03-01_20-26-47_15_1.mp4',  # 摄像头太暗 FP close eye
'dump_2135857511_15_1.mp4',  # 摄像头太暗 FP
'dump_783466853_15_1.mp4',  # 摄像头太暗 FP
'dump_1032485638_15_1.mp4',  # 摄像头太暗 FP
'dump_1986115433_15_1.mp4',  # 摄像头太暗 FP
'dump_2439252446_15_1.mp4',  # 摄像头太暗 FP
'dump_3432526067_15_1.mp4',  # 摄像头太暗 FP
'dump_3601563961_15_1.mp4',  # 摄像头太暗 FP
'Video_2023-03-04_10-22-16_15_1.mp4',  # 低头闭眼 FP
'dump_20230305141629_15_1.mp4',  # 低头闭眼 FP
'dump_20230523190337_15_1.mp4',  # 摄像头太暗 FP
'dump_20230524191649_15_1.mp4',  # 摄像头太暗 FP
'dump_20230524193021_15_1.mp4',  # 摄像头太暗 FP
'dump_3790894981_15_1.mp4',  # 摄像头太暗 FP
'dump_20230322095054_15_1.mp4',  #mojing FN
'dump_20230305140208_15_1.mp4', #no close eye FN
'dump_878868486_15_1.mp4', #no close eye FN
'dump_20230523193215_15_1.mp4', #摄像头太暗 FN
'dump_777534852_15_1.mp4',#摄像头太暗 FN
'dump_789127592_15_1.mp4',  # 摄像头太暗 FN
'dump_800849691_15_1.mp4',  # 摄像头太暗 FN
'dump_20230523185910_15_1.mp4', # 摄像头太暗 FN
"Video_2023-03-01_20-42-49_15_1.mp4", # 摄像头太暗 FN
# --------集度表格备注删除-----
"dump_3004977857_15_1.mp4",
"dump_3023655255_15_1.mp4",
"dump_3037338495_15_1.mp4",
"dump_3065700584_15_1.mp4",
"dump_3078345080_15_1.mp4",
"dump_3091339522_15_1.mp4",
"dump_3250585051_15_1.mp4",
"dump_3280029064_15_1.mp4",
"dump_3293869730_15_1.mp4",
"dump_3332429370_15_1.mp4",
"dump_3349226095_15_1.mp4",
"dump_1899614735_15_1.mp4",
"dump_1993625633_15_1.mp4",
"dump_20230322101321_15_1.mp4",
"dump_20230322101342_15_1.mp4",
"dump_20230323121122_15_1.mp4",
"dump_20230323121136_15_1.mp4",
"dump_20230323121201_15_1.mp4",
"dump_20230323124426_15_1.mp4",
"Video_2023-03-03_11-15-13_15_1.mp4",
"Video_2023-03-03_11-17-05_15_1.mp4",
"dump_20230323181302_15_1.mp4",
"dump_20230323181323_15_1.mp4",
"dump_20230323181354_15_1.mp4",
"Video_2023-03-04_10-24-24_15_1.mp4",
"Video_2023-03-04_10-25-37_15_1.mp4",
"Video_2023-03-06_16-50-25_15_1.mp4",
"Video_2023-03-06_16-51-41_15_1.mp4",
"Video_2023-03-06_16-52-00_15_1.mp4",
"Video_2023-03-06_16-52-49_15_1.mp4",
"Video_2023-03-06_16-53-00_15_1.mp4",
"Video_2023-03-06_16-53-17_15_1.mp4",
"dump_20230305140125_15_1.mp4",
"dump_20230305140313_15_1.mp4",
"dump_20230411124545_15_1.mp4",
"dump_20230411124604_15_1.mp4",
"dump_20230411125801_15_1.mp4",
"dump_20230523185940_15_1.mp4",
"dump_20230523191530_15_1.mp4",
"dump_20230523191546_15_1.mp4",
"dump_20230524191223_15_1.mp4",
"dump_20230524191245_15_1.mp4",
"dump_20230524191940_15_1.mp4",
"Video_2023-03-06_18-53-57_15_1.mp4",
"Video_2023-03-06_18-54-21_15_1.mp4",
"Video_2023-03-06_18-54-38_15_1.mp4",
"Video_2023-03-06_18-54-49_15_1.mp4",
"Video_2023-03-06_18-55-15_15_1.mp4",
"Video_2023-03-06_18-55-26_15_1.mp4",
"Video_2023-03-06_18-55-37_15_1.mp4",
"dump_20230320193829_15_1.mp4",
"dump_20230320201705_15_1.mp4",
"Video_2023-03-13_18-45-05_15_1.mp4",
"Video_2023-03-13_18-45-15_15_1.mp4",
"Video_2023-03-13_18-45-27_15_1.mp4",
"Video_2023-03-13_18-46-48_15_1.mp4",
"Video_2023-03-13_18-47-04_15_1.mp4",
"Video_2023-03-13_18-47-40_15_1.mp4",
"Video_2023-03-13_18-47-54_15_1.mp4",
"Video_2023-03-13_18-48-22_15_1.mp4",
"Video_2023-03-13_18-48-32_15_1.mp4",
"dump_1345644960_15_1.mp4",
"dump_1409811188_15_1.mp4",
"dump_1425889804_15_1.mp4",
"dump_1524972359_15_1.mp4",
"dump_1569651522_15_1.mp4",
"dump_1622934425_15_1.mp4",
"dump_1694297625_15_1.mp4",
"dump_1734349162_15_1.mp4",
"dump_1756897706_15_1.mp4",
"dump_1772350103_15_1.mp4",
"dump_1877303284_15_1.mp4",
"dump_1900061987_15_1.mp4",
"dump_1919706797_15_1.mp4",
"dump_900494594_15_1.mp4",
"dump_919859475_15_1.mp4",
"dump_996973602_15_1.mp4",
"Video_2023-03-01_20-40-57_15_1.mp4",
"Video_2023-03-01_20-41-17_15_1.mp4",
"dump_20230522202118_15_1.mp4",
"dump_20230522202436_15_1.mp4",
"dump_20230522203822_15_1.mp4",
"dump_20230522203906_15_1.mp4"
]


                   #  "00-01-00-00-09/dump_20230323111345_15_1",  # close eye
                   # "00-01-00-01-09/dump_20230323112853_15_1",  # close eye
                   # "00-01-00-01-09/dump_20230323112941_15_1",  # close eye
                   #  "01-06-01-02-06/dump_20230524193047_15_1",  # 低头闭眼
                   # "01-15-00-00-06/dump_20230323195445_15_1",  # 低头闭眼
                   # "01-18-01-00-00/dump_20230525191354_15_1",  #不打哈欠时闭眼检出
                   # "01-18-01-00-06/dump_20230525191848_15_1",  # 转头前闭眼检出
                   #  "05-11-01-02-06/dump_20230522210720_15_1",  # 低头闭眼
                   #  "05-11-01-00-00/dump_20230522205135_15_1",  #打哈欠结束后闭眼检出
                   #  "03-01-01-00-06/dump_136418351_15_1",   # 低头闭眼
                   #  "00-11-01-00-00/dump_20230305143929_15_1",   # 低头闭眼
                   # "00-11-01-00-00/dump_20230305144336_15_1",   # 低头闭眼


    #                "dump_20230305142729_15_1",
    #                "dump_20230305145040_15_1",
    #                "dump_20230323195702_15_1"

    # ignore_list = ["00-00-01-00-00/dump_2185824757_15_1",
    #                "00-02-00-00-06/dump_4376520089_15_1",
    #                "00-02-04-01-09/dump_20230322102900_15_1",
    #                "00-03-00-00-09/dump_20230323120705_15_1",
    #                "00-03-00-01-08/dump_20230323122245_15_1",
    #                "00-03-00-01-09/dump_20230323122342_15_1",
    #                 "00-04-01-00-09/dump_20230323124058_15_1",
    #                "00-04-01-01-09/dump_20230323125453_15_1",
    #                "00-06-01-00-00/Video_2023-03-03_11-07-00_15_1",
    #                "00-06-01-00-06/Video_2023-03-03_11-48-07_15_1",
    #                "00-06-01-00-08/dump_20230323180811_15_1",
    #                "00-06-01-00-09/dump_20230323180927_15_1",
    #                "00-06-01-01-09/dump_20230323182441_15_1",
    #                 "00-08-01-01-06/dump_20230411122433_15_1",
    #                "00-08-01-01-07/dump_20230411122541_15_1",
    #                "00-08-01-01-09/dump_20230411122750_15_1",
    #                 "00-10-00-00-02/dump_20230305141629_15_1",
    #                "00-10-00-00-06/dump_20230305142729_15_1",
    #                "00-11-01-00-00/dump_20230305143929_15_1",
    #                 "00-11-01-00-00/dump_20230305144336_15_1",
    #                 "00-11-01-00-02/dump_20230305145040_15_1",
    #                 "00-11-01-00-07/dump_20230323135822_15_1",
    #                "00-11-01-00-09/dump_20230323140111_15_1",
    #                "00-11-01-01-09/dump_20230323141409_15_1",
    #                "00-12-00-00-06/dump_20230305165720_15_1",
    #                "00-18-01-00-00/dump_20230324113549_15_1",
    #                "00-18-01-00-06/dump_20230324114630_15_1",
    #                "00-18-01-00-09/dump_20230324115158_15_1",
    #                "00-18-01-01-00/dump_20230324115435_15_1",
    #                "00-18-01-01-09/dump_20230324120542_15_1",
    #                 "01-04-01-00-07/dump_20230523190604_15_1",
    #                "01-06-01-00-06/dump_20230524191649_15_1",
    #                "01-06-01-02-06/dump_20230524193047_15_1",
    #                "01-08-01-02-06/dump_20230523194417_15_1",
    #                "01-10-00-01-06/dump_20230413200001_15_1",
    #                "01-10-00-01-09/dump_20230413200537_15_1",
    #                "01-12-00-00-00/dump_20230524195809_15_1",
    #                "01-15-00-00-00/dump_20230323193822_15_1",
    #                 "01-15-00-00-06/dump_20230323195052_15_1",
    #                "01-15-00-00-06/dump_20230323195445_15_1",
    #                "01-15-00-00-07/dump_20230323195702_15_1",
    #                "01-15-00-00-09/dump_20230323200103_15_1",
    #                "01-15-00-01-09/dump_20230323201555_15_1",
    #                "01-18-01-00-00/dump_20230525191354_15_1",
    #                 "01-25-01-00-06/dump_20230525194332_15_1",
    #                "02-05-01-00-09/dump_19700101083747_15_1",
    #                "02-05-01-00-09/dump_19700101085513_15_1",
    #                "02-08-01-00-09/dump_19700101082455_15_1",
    #                "02-12-00-00-09/dump_19700101085644_15_1",
    #                "02-12-00-01-07/dump_19700101080521_15_1",
    #                "02-12-00-01-09/dump_19700101080711_15_1",
    #                "03-01-01-00-06/dump_136418351_15_1",
    #                "03-01-01-00-06/dump_392830147_15_1",
    #                "04-00-01-01-06/dump_20230320202455_15_1",
    #                "04-03-00-00-00/dump_1032485638_15_1",
    #                "04-03-00-00-02/dump_1986115433_15_1",
    #                "04-03-00-00-02/dump_2439252446_15_1",
    #                 "04-03-00-00-06/dump_3432526067_15_1",
    #                "04-03-00-00-06/dump_3601563961_15_1",
    #                "04-03-00-00-06/dump_3795545751_15_1",
    #                "04-04-01-00-00/dump_405848865_15_1",
    #                "04-04-01-00-00/dump_590839890_15_1",
    #                "04-04-01-00-06/dump_1931246427_15_1",
    #                "04-04-01-00-06/dump_2396420827_15_1",
    #                "05-05-01-00-00/Video_2023-03-01_20-26-47_15_1",
    #                "05-05-01-00-00/Video_2023-03-01_20-30-40_15_1",
    #                "05-05-01-00-02/Video_2023-03-01_20-45-47_15_1",
    #                "05-05-01-00-06/Video_2023-03-01_21-08-26_15_1",
    #                "05-05-01-00-06/Video_2023-03-01_21-13-58_15_1",
    #                "05-10-00-00-08/dump_20230522203252_15_1",
    #                "05-10-00-02-06/dump_20230522204116_15_1",
    #                "05-11-01-00-00/dump_20230522205135_15_1",
    #                "05-11-01-02-06/dump_20230522210720_15_1",
    #                "00-01-00-00-09/dump_20230323111345_15_1",
    #                "00-01-00-01-09/dump_20230323112853_15_1",
    #                "00-01-00-01-09/dump_20230323112941_15_1",
    #                "00-02-00-00-00/dump_1612847612_15_1",
    #                "00-02-04-00-09/dump_20230322100928_15_1",
    #                "05-11-01-00-06/dump_20230522205526_15_1", # hei
    #                 "05-10-00-00-07/dump_20230522203213_15_1", # hei
    #                 "05-10-00-00-06/dump_20230522202801_15_1", #hei
    #                "04-03-00-00-00/dump_739515576_15_1", #biyan
    #                ]
    test = []

    fw_fp = open(events_result_path+'/fp.txt', 'w')
    fw_fn = open(events_result_path+'/fn.txt', 'w')

    files = glob.glob(events_result_path + '/result*.txt')
    files.sort()
    # thresholds = np.linspace(3.0, 5.0, 21)
    # thresholds = np.linspace(0.5, 0.65, 16)
    thresholds = [0.5]
    global video_fps, check_fps

    for thresh in thresholds:
        print(thresh)
        total = 0
        count_have_report_event = 0
        tp = fp = tn = fn = 0
        for event_result_file in files:

            if 'oms_fatigue_eyeClosing_test_20240102' in event_result_file:
                # video_dir_oms = "/media/baidu/117da259-fdcf-4eef-8f7d-e1be47008443/集度提供的15%测试集231018/eye/wangyawei/oms_fatigue_eyeClosing_test_20231013/"
                video_name_ = os.path.basename(event_result_file).split('__')[2].split('dms_fatigue_eyeClosing_test_20231012_')[-1].split('oms_fatigue_eyeClosing_test_20240102_')[-1].split('.mp4')[0]
                if '_Video' in video_name_:
                    video_name = video_name_.replace('_Video', '/Video')
                elif '_dump' in video_name_:
                    video_name = video_name_.replace('_dump', '/dump')
            # video_path = video_dir_oms + video_name + '.mp4'
            # print('video_path', video_path)
                video_name = video_name + '.mp4'
                video_name = video_name.split('/')[-1]
                print('video_name', video_name)

                if video_name in ignore_list:  # recall: 87.59%, precision: 60.48% --> recall: 87.59%, precision: 61.35%
                    continue

            pred_results = read_pred_file(event_result_file)

            if 'dms_fatigue_eyeClosing_test_20231012' in event_result_file:
                label_class = os.path.basename(event_result_file).split('__')[2].split('dms_fatigue_eyeClosing_test_20231012_')[-1].split('_')[0].split('-')[-1]
            elif 'oms_fatigue_eyeClosing_test_20240102' in event_result_file:
                label_class = os.path.basename(event_result_file).split('__')[2].split('oms_fatigue_eyeClosing_test_20240102_')[-1].split('_')[0].split('-')[-1]
            else:
                print('Error label_class of event_result_file!!!')

            label_closeeye = 0
            if label_class == '01':
                label_closeeye = 1

            close_eye_window_event = []
            report_frames = []
            have_report_event = 0
            check_count = 0
            # frame_count = 0
            for pred in pred_results:
                key = pred['key']
                frame_count = int(key.replace('frame_', ''))

                if if_need_sample:
                    # check [0, 3, 5, 8]
                    mod_ind = frame_count % 10
                    # frame_count += 1
                    if mod_ind not in [0, 3, 5, 8]:
                        continue

                if len(pred) < 3:  # no face
                    eye_close = 0
                else:
                    emo_class = pred['emo_class']
                    emo_prob = float(pred['emo_prob'])
                    # print('pred', pred)
                    close_result = is_close_eye(pred,
                                                thresh_if_eye=0.5,
                                                OPEN_CALSS=OPEN_CALSS,
                                                emo_class=emo_class,
                                                emo_prob=emo_prob
                                                )
                    eye_close = close_result[0]

                close_eye_window_event.append(eye_close)
                report_close_eye = 0
                close_ratio = 0
                if len(close_eye_window_event) > window_length:
                    close_eye_window_event.pop(0)
                    close_ratio = sum(close_eye_window_event) / len(close_eye_window_event)
                    if close_ratio >= 0.8:
                        have_report_event = 1
                        report_close_eye = 1
                        report_frames.append(key)
            is_fp = is_fn = is_tp = is_tn = 0
            if have_report_event:
                count_have_report_event += 1
                if label_closeeye == 1:
                    is_tp = 1
                    tp += 1
                else:
                    is_fp = 1
                    fp += 1
            else:
                if label_closeeye == 1:
                    is_fn = 1
                    fn += 1
                else:
                    is_tn = 1
                    tn += 1
            total += 1
            if is_fp:
                print(os.path.basename(event_result_file), 'pred_frames:', len(pred_results), 'fp')
                if 'dms_fatigue_eyeClosing_test_20231012' in event_result_file:
                    video_path_ = event_result_file.split('dms_fatigue_eyeClosing_test_20231012_')[-1].split('.mp4')[0]
                else:
                    video_path_ = event_result_file.split('oms_fatigue_eyeClosing_test_20240102_')[-1].split('.mp4')[0]
                video_dir = video_path_.split('_')[0]
                video_name = video_path_.split(video_dir)[-1]
                video_path = (video_dir+'/'+video_name+'.mp4').replace('/_', '/')
                fw_fp.write(video_path + '\n')

            if is_fn:
                # print(os.path.basename(event_result_file), 'pred_frames:', len(pred_results), 'fn')
                if 'dms_fatigue_eyeClosing_test_20231012' in event_result_file:
                    video_path_ = event_result_file.split('dms_fatigue_eyeClosing_test_20231012_')[-1].split('.mp4')[0]
                else:
                    video_path_ = event_result_file.split('oms_fatigue_eyeClosing_test_20240102_')[-1].split('.mp4')[0]
                video_dir = video_path_.split('_')[0]
                video_name = video_path_.split(video_dir)[-1]
                video_path = (video_dir+'/'+video_name+'.mp4').replace('/_', '/')
                fw_fn.write(video_path + '\n')

        print(events_result_path)
        print('videos total: %d, close_eye_report: %d' % (total, count_have_report_event))
        print('tp: %d, fp: %d, fn: %d, tn: %d' % (tp, fp, fn, tn))
        print('recall: %.2f%%, precision: %.2f%%' % (float(tp) / (tp + fn) * 100, float(tp) / (tp + fp) * 100))
        print('check', len(sorted(list(set(test)))), sorted(list(set(test))))

    fw_fp.close()
    fw_fn.close()
    return


def read_pred_file(pred_file):
    """read_pred_file, txt, json lines."""
    with open(pred_file, 'r') as f:
        lines = f.readlines()
    pred_results = []
    for line in lines:
        pred_dict = json.loads(line.strip())
        if len(pred_dict) == 0:
            continue
        pred_results.append(pred_dict)
    return pred_results


def read_label_file(label_file):
    """read_label_file, txt, json lines."""
    with open(label_file, 'r') as f:
        lines = f.readlines()
    gt_labels = {}
    for line in lines:
        line_sp = line.strip().split('\t')
        key = line_sp[0]
        status_dict = json.loads(line_sp[1])
        eye_status = status_dict['eyeState']  # eyeState: 1: open eye, 2: close eye
        gt_labels[key] = eye_status - 1
    return gt_labels


def eval_dataset(events_paths, is_videos, label, pred_result_path, fuse_bn_1layer, OPEN_CALSS):
    """eval_dataset, inference and count acc."""
    global total, check_count, count, count_left, count_right
    if not os.path.exists(pred_result_path):
        os.makedirs(pred_result_path)

    print(events_paths)
    for root, dirs, files in os.walk(events_paths):
        for i, fie in enumerate(files):
            if fie.startswith('._'):
                continue
            if fie.endswith('mp4') or fie.endswith('avi') or fie.endswith('jpg'):
                event_path = os.path.join(root, fie)
                if event_path.endswith('mp4') or event_path.endswith('jpg'):
#                     result_file = os.path.join(pred_result_path, 'result.txt')
                    result_file = os.path.join(pred_result_path, 'result__%s__%s.txt' % ('_'.join(events_path.split('/')[-2:]),
                                                                     '_'.join(event_path.split('/')[-4:])))
                    f_out = open(result_file, 'w')
                    file_path = event_path
                    cap = cv.VideoCapture(file_path)
                    print('video file:', file_path)
                    print('video size(w, h, c):', int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                          int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
                          int(cap.get(cv.CAP_PROP_CHANNEL)))
                    print('video frame count:', int(cap.get(cv.CAP_PROP_FRAME_COUNT)))
                    frame_ind = 0
                    while 1:
                        result, frame = cap.read()
                        if not result:
                            break
                        frame_name = 'frame_%05d.jpg' % frame_ind
                        # print('frame_name', frame_name)
                        cur_ind = frame_ind
                        frame_ind += 1
                        total += 1
                        # if cur_ind % 10 not in [0, 3, 5, 8]:
                        #     continue
                        # print('[frame_ind]', frame_ind)
                        imgfile_path = os.path.join(event_path[:-4], frame_name)
                        pred_results = deal_single_image(frame, imgfile_path, OPEN_CALSS, fuse_bn_1layer, pred_result_path)
                        if 'ignored' in pred_results:
                            continue
                        else:
                            if len(pred_results) == 0:
                                print('no face!!!', imgfile_path)
                            else:
                                print(imgfile_path, pred_results['eye_close'])
                        pred_results['key'] = frame_name[:-4]
                        for k, v in pred_results.items():
                            if isinstance(v, np.float32) or isinstance(v, np.float64):
                                pred_results[k] = '%.6f' % v
                        json_str = json.dumps(pred_results)
                        f_out.write(json_str + '\n')
                    f_out.close()
    return


if __name__ == "__main__":
    root_dir = "/home/baidu/Desktop/chengq/data/eyeCenter_model"
    date = "0722"
    
    # 人脸检测模型
    model_path_facedetect = os.path.join(root_dir, "FaceDetection20240301V7Main_noquant.onnx")
    facedetect = FaceDetectOnnx(pathModel=model_path_facedetect)
    # 人脸关键点模型
    onnx_facept = os.path.join(root_dir, "FacialLandmark230403V26Main_noquant.onnx")
    ort_sess_facept = onnxruntime.InferenceSession(onnx_facept)
    # 人脸表情模型
    onnx_emo = os.path.join(root_dir, "FaceExpression240117V7MainQAT_sim_skipbn.onnx")
    EmoNet = onnxruntime.InferenceSession(onnx_emo)
    # 瞳孔检测模型
    onnx_facepupil = os.path.join(root_dir, date, f"file_onnx/eyecenter_{date}_sim_skipbn.onnx")
    ort_sess_pupil = onnxruntime.InferenceSession(onnx_facepupil)
    
    net_inputH = 192
    net_inputW = 192
    total = 0
    check_count = 0
    count = 0
    count_left = 0
    count_right = 0
    close_eye_window = []
    check_fps = 24  # use [0,3,5,8] frame_ind as 10 fps.
    video_fps = 24
    # window_length = round(check_fps * 1.5)  # 1s
    window_length = round(10 * 1.5)  # 1s
    is_oms = 0  # 0: dms, 1: oms
    is_oms_left = 1  # 0: right person, 1: left person, 2: either
    draw_image = 0  # if draw_image and show
    save_result = 0  # save draw image
    is_videos = 0  # 0: images, 1: videos
    label = 1  # 0: no close eye in video, 1: close eye in video  FN:1 FP:0
    fuse_bn_1layer = True #模型合并bn和首层使用True, 没有合并使用False
    OPEN_CALSS = 2

    # 0) QA test
    # events_path = '/media/baidu/6bb5a831-f973-4b07-9b64-2d868003e832/QA/1/eyestate_copy/dms'
    # is_oms = 0
    # """
    # close_img, close2close, close2open, recall, precision 1665 1642 23 0.9861861855938822 0.9767995235117194
    # open_img, open2open, open2close, recall, precision 1981 1942 39 0.9803129727509777 0.9882951648914527
    # """
    # events_path = '/media/baidu/6bb5a831-f973-4b07-9b64-2d868003e832/QA/1/eyestate_copy/oms'
    # is_oms = 0  # QA测试, oms推理使用0，因为人只有1个可能坐主驾也可能坐副驾
    # """
    # close_img, close2close, close2open, recall, precision 1371 1269 102 0.9256017498719169 0.9628224575395884
    # open_img, open2open, open2close, recall, precision 1269 1220 49 0.9613869180761332 0.9228441747936125
    #
    # """

    # 1)集度提供的15%测试集:
    # events_path = '/media/baidu/117da259-fdcf-4eef-8f7d-e1be47008443/集度提供的15%测试集231018/eye/wangyawei/15/dms_fatigue_eyeClosing_test_20231012'
    # is_oms = 0
    # events_path = '/media/baidu/117da259-fdcf-4eef-8f7d-e1be47008443/集度提供的15%测试集231018/eye/wangyawei/15/oms_fatigue_eyeClosing_test_20240102'
    # is_oms = 1
    
    # 2) bug:
    # events_path = '/media/baidu/117da259-fdcf-4eef-8f7d-e1be47008443/Badcase_eye/bug/zmj/240125_FP/oms/oms_eyeclose_sensing-v1.1.0.26-20240125_FP-demo' #0301err7 0303err8 xxx
    # is_oms = 1
    # events_path = '/media/baidu/117da259-fdcf-4eef-8f7d-e1be47008443/Badcase_eye/bug/zmj/240221_FP/0221_FP'#0322err3 #0301err0 0303err2 xxx0301
    # is_oms = 0
    # events_path = '/media/baidu/117da259-fdcf-4eef-8f7d-e1be47008443/Badcase_eye/bug/zmj/240227_FP/0227FP/video'#0322err1  #0301err1     0303err1 xxx
    # is_oms = 0

    # events_path = '/media/baidu/117da259-fdcf-4eef-8f7d-e1be47008443/Badcase_eye/bug/zmj/231211/video_all/20231208_oms闭眼_FN/video/close' #ok
    # [eyeCenter_ft_240118_prun50_clsloss100_epoch96_sim_skipbn] videos total: 9, close_eye_report: 9 (0.45)
    # [eyeCenter_ft_240118_prun50_clsloss100_epoch96_sim_skipbn] videos total: 9, close_eye_report: 9 (0.4)
    # is_oms = 1
    # events_path = '/media/baidu/117da259-fdcf-4eef-8f7d-e1be47008443/Badcase_eye/bug/zmj/231211/video_all/20231208_oms闭眼_FP/video/open' #err5
    # [eyeCenter_ft_240118_prun50_clsloss100_epoch96_sim_skipbn] (3bujie) videos total: 11, close_eye_report: 5(0.45)
    # [eyeCenter_ft_240118_prun50_clsloss100_epoch96_sim_skipbn] (3bujie) videos total: 11, close_eye_report: 5(0.4)
    # is_oms = 1
    # events_path = '/media/baidu/117da259-fdcf-4eef-8f7d-e1be47008443/Badcase_eye/bug/zmj/231211/video_all/wubao/open/'#err5!
    # [eyeCenter_ft_240118_prun50_clsloss100_epoch96_sim_skipbn] videos total: 41, close_eye_report: 2(mouth18) / 5(mouth28)  1 has close
    # 'FP_1205_video_err_00004.mp4' / 百度1205版本误报视频_has_close_cameradump_20230906133211_14_5_fn13 (cover)
    # is_oms = 0
    # events_path = "/media/baidu/117da259-fdcf-4eef-8f7d-e1be47008443/Badcase_eye/bug/zmj/231211/video_all/loubao/" #ok
    # [eyeCenter_ft_240118_prun50_clsloss100_epoch96_sim_skipbn] videos total: 6, close_eye_report: 6
    # is_oms = 0
    # events_path = "/media/baidu/PortableSSD/11111/eye240121"  # demo
    # is_oms = 0
    # events_path = "/media/baidu/117da259-fdcf-4eef-8f7d-e1be47008443/Badcase_eye/bug/zmj/240313_FP/0313FP/video"
    # is_oms = 0
    # events_path = "/media/baidu/117da259-fdcf-4eef-8f7d-e1be47008443/Badcase_eye/bug/zmj/240703_FP/0703FP"
    # is_oms = 0
    events_path = "/media/baidu/117da259-fdcf-4eef-8f7d-e1be47008443/Badcase_eye/bug/zmj/240713_FP/0713FP"
    is_oms = 0

    # 3) demo.jpg
    # events_path = "/media/baidu/17d80214-bfce-43ec-807d-d6ba4e44dfd6/EyeCenter/duiqi"  # demo
    # is_oms = 0

    if "QA" in events_path:
        dataset_flag = "QA"
    elif "集度" in events_path:
        dataset_flag = "集度"
    elif "bug" in events_path:
        dataset_flag = "bug"
    else:
        dataset_flag = "duiqi"
    pred_result_path = os.path.join(root_dir, 
                                    date, 
                                    "results", 
                                    onnx_facepupil.split('/')[-1].replace('.onnx', ''), 
                                    dataset_flag,
                                    events_path.split("/")[-1])
    eval_dataset(events_path, is_videos, label, pred_result_path, fuse_bn_1layer, OPEN_CALSS) # 模型推理
    # eval_by_json_QA(pred_result_path, OPEN_CALSS=OPEN_CALSS)  # 出指标:BaiduQA测试集
    # eval_jidu_testdata_by_json_file(pred_result_path, if_need_sample=1, OPEN_CALSS=OPEN_CALSS) # 出指标:jidu15%测试集
    eval_by_json_file(pred_result_path, if_need_sample=0, OPEN_CALSS=OPEN_CALSS) # bug
