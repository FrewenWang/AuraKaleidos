#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
predict images
"""
# import paddle
import os, shutil
import time
import json
import cv2
import numpy as np
import random
import math
from PIL import Image
from scipy.linalg import solve
# from imgaug import augmenters as iaa
# from paddleslim.dygraph import L1NormFilterPruner
# from paddleslim.dygraph.quant import QAT


import facedetect_onnx_220114
import face_landmark_220324


landmark106_new = [
                (3, 39),
                (4, 52),
                (6, 64),
                (9, 77),
                (12, 90),
                (16, 102),
                (20, 113),
                (25, 122),
                (31, 131),
                (36, 137),
                (42, 144),
                (49, 148),
                (55, 153),
                (61, 156),
                (67, 159),
                (76, 160),
                (81, 161),
                (86, 160),
                (95, 159),
                (100, 156),
                (106, 153),
                (112, 148),
                (119, 144),
                (125, 137),
                (131, 131),
                (137, 122),
                (141, 113),
                (145, 102),
                (150, 90),
                (153, 77),
                (156, 64),
                (157, 52),
                (159, 39),
                (10, 19),
                (20, 10),
                (33, 9),
                (48, 12),
                (61, 17),
                (22, 20),
                (35, 20),
                (48, 20),
                (61, 20),
                (100, 17),
                (113, 12),
                (128, 9),
                (141, 10),
                (151, 19),
                (100, 20),
                (113, 20),
                (127, 20),
                (140, 20),
                (26, 39),
                (35, 36),
                (45, 36),
                (55, 42),
                (45, 44),
                (35, 44),
                (41, 36),
                (39, 44),
                (41, 39),
                (41, 41),
                (106, 42),
                (116, 36),
                (127, 36),
                (135, 39),
                (127, 44),
                (116, 44),
                (121, 36),
                (122, 44),
                (121, 39),
                (121, 41),
                (81, 38),
                (81, 55),
                (81, 73),
                (81, 89),
                (70, 42),
                (93, 41),
                (63, 79),
                (100, 77),
                (54, 86),
                (109, 86),
                (65, 93),
                (71, 97),
                (81, 100),
                (92, 97),
                (97, 93),
                (55, 124),
                (74, 127),
                (81, 125),
                (89, 127),
                (108, 124),
                (81, 138),
                (64, 125),
                (99, 125),
                (99, 131),
                (64, 131),
                (57, 124),
                (106, 124),
                (71, 122),
                (81, 122),
                (92, 122),
                (71, 127),
                (81, 128),
                (92, 127),
                (76, 137),
                (87, 137)]

class DMSDanger(object): #GetValidDatasets

    def landmarks_call_train_bigRect(self):
        cor_landmark106 = np.array(landmark106_new)
        face_w, face_h = cor_landmark106.max(axis=0) - cor_landmark106.min(axis=0)
        x_left = (cor_landmark106[52][0] + cor_landmark106[53][0] + cor_landmark106[55][0] + cor_landmark106[56][0]) / 4
        y_left = (cor_landmark106[52][1] + cor_landmark106[53][1] + cor_landmark106[55][1] + cor_landmark106[56][1]) / 4
        x_right = (cor_landmark106[62][0] + cor_landmark106[63][0] + cor_landmark106[65][0] + cor_landmark106[66][
            0]) / 4
        y_right = (cor_landmark106[62][1] + cor_landmark106[63][1] + cor_landmark106[65][1] + cor_landmark106[66][
            1]) / 4
        w_eye = math.sqrt((x_left - x_right) ** 2 + (y_left - y_right) ** 2)
        crop_w = int(3. * w_eye / 2. + 0.5)
        crop_h = int(3. * crop_w / 2 + 0.5)

        right_start = (np.array([cor_landmark106[75][0] - (1.62 * crop_w + 1), cor_landmark106[75][1] - 0.1 * crop_h]) + 0.5).astype(np.int64)
        left_start = (np.array([cor_landmark106[76][0], cor_landmark106[76][1] - 0.1 * crop_h]) + 0.5).astype(np.int64)

        cor_landmark106_right = cor_landmark106 - right_start
        cor_landmark106_left = cor_landmark106 - left_start

        return cor_landmark106_right, cor_landmark106_left, int(0.5 + crop_w * 1.62 + 1), int(0.5 + crop_h * 1.2)


    def alignWithLandmark(self, image, pts, landmark106, face_w_2, face_h_2):
        """crop"""
        M = self.align1(pts, landmark106)
        cor_img = cv2.warpAffine(image, M, (face_w_2, face_h_2))
        cor_pts = np.dot(M, np.column_stack((pts, [1] * pts.shape[0])).T).T
        cor_pts = cor_pts[:, :2]
        return cor_img, cor_pts

    def align1(self, X1, X2):
        """最小二乘视线矩阵对齐"""
        x11, x12 = X1[:, 0], X1[:, 1]
        y11, y12 = X2[:, 0], X2[:, 1]

        M01 = np.dot(x11, x11.T) + np.dot(x12, x12.T)
        M02 = 0
        M03 = np.sum(x11)
        M04 = np.sum(x12)
        M05 = (np.dot(y11, x11.T) + np.dot(x12, y12.T)) * -1

        M11 = 0
        M12 = M01
        M13 = M04 * -1
        M14 = M03
        M15 = np.dot(y11, x12.T) - np.dot(x11, y12.T)

        M21 = M03
        M22 = M04 * -1
        M23 = 106
        M24 = 0
        M25 = np.sum(y11) * -1

        M31 = M04
        M32 = M03
        M33 = 0
        M34 = 106
        M35 = np.sum(y12) * -1

        M = np.array([M01, M02, M03, M04, M11, M12, M13, M14, M21, M22, M23, M24, M31, M32, M33, M34]).reshape(4, 4)
        x1 = solve(M, np.array([M05 * -1, M15 * -1, M25 * -1, M35 * -1]))
        x = np.array([x1[0], x1[1] * -1, x1[2], x1[1], x1[0], x1[3]]).reshape(2, 3)
        return x


    def show(self, frame_copy, DMSType, directionIsLeft):
        """
        0: normal  1:smoke  2:silence  3:drink  4: open  5:mask  6:捂嘴(挡嘴)  7.有手(不挡嘴)
        """
        if directionIsLeft == 1:
            x1, y0 = 200, 200
            cv2.putText(frame_copy, "left:", (int(x1-100), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            x1, y0 = 200, 300
            cv2.putText(frame_copy, "right:", (int(x1-100), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if DMSType == 1:
            cv2.putText(frame_copy, "call", (int(x1), int(y0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame_copy, "normal", (int(x1), int(y0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    def load_pretrain_weight(self, model, weight):
        """
        加载预训练模型
        :param model:
        :param weight:
        :return:
        """
        weights_path = weight + '.pdparams'

        if not (os.path.exists(weights_path)):
            raise ValueError("Model pretrain path `{}` does not exists. "
                             "If you don't want to load pretrain model, "
                             "please delete `pretrain_weights` field in "
                             "config file.".format(weights_path))

        model_dict = model.state_dict()

        param_state_dict = paddle.load(weights_path)
        ignore_weights = set()

        for name, weight in param_state_dict.items():
            if name in model_dict.keys():
                if list(weight.shape) != list(model_dict[name].shape):
                    ignore_weights.add(name)
            else:
                ignore_weights.add(name)

        for weight in ignore_weights:
            param_state_dict.pop(weight, None)

        model.set_dict(param_state_dict)

    def mat_inter(self, box1, box2):
        """判断两个矩形是否相交 box=(xA,yA,xB,yB)"""
        #print('box1, box2', box1, box2)

        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2

        lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
        ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
        sax = abs(x01 - x02)
        sbx = abs(x11 - x12)
        say = abs(y01 - y02)
        sby = abs(y11 - y12)
        if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
            return True
        else:
            return False

    def solve_coincide(self, box1, box2):
        """box=(xA,yA,xB,yB) 计算两个矩形框的重合度"""
        if self.mat_inter(box1, box2) == True:
            x01, y01, x02, y02 = box1
            x11, y11, x12, y12 = box2

            col = min(x02, x12) - max(x01, x11)
            row = min(y02, y12) - max(y01, y11)
            intersection = col * row
            area1 = (x02 - x01) * (y02 - y01)
            area2 = (x12 - x11) * (y12 - y11)
            coincide = float(intersection) / float((area1 + area2) - intersection)
            # coincide = intersection / area2
            return coincide
        else:
            return 0


    def non_max_suppression_slow(self, boxes, overlapThresh, max_hand_num):
        """
        非极大值抑制
        """
        suppress = []
        if len(boxes) == 0:
            return []
        if len(boxes) > 1:
            for i in range(len(boxes)):
                if i == 0:
                    overlap = self.solve_coincide(boxes[i], boxes[i + 1])
                    if overlap > overlapThresh:
                        x1, y1, x2, y2 = boxes[i]
                        x10, y10, x20, y20 = boxes[i + 1]
                        x = min(x1, x10)
                        y = min(y1, y10)
                        x0 = max(x2, x20)
                        y0 = max(y2, y20)
                        box = [x, y, x0, y0]
                        if len(suppress) < max_hand_num:
                            suppress.append(box)  # 前2个框是1张人脸
                    else:  # 前2个框不是1张人脸
                        if len(suppress) < max_hand_num:
                            suppress.append(boxes[i])
                        if len(suppress) < max_hand_num:
                            suppress.append(boxes[i + 1])
                # print "suppress1", suppress
                if i > 1:
                    for k in range(len(suppress)):  # 每个框和suppress对比
                        overlap = self.solve_coincide(boxes[i], suppress[k])
                        if overlap > overlapThresh:
                            x1, y1, x2, y2 = boxes[i]
                            x10, y10, x20, y20 = suppress[k]
                            x = min(x1, x10)
                            y = min(y1, y10)
                            x0 = max(x2, x20)
                            y0 = max(y2, y20)
                            box = [x, y, x0, y0]
                            suppress.append(box)  # 是1张人脸，取最大值, 替换
                            suppress.pop(k)
                            # print "suppress2", suppress
                        else:  # 不是1张人脸，和suppress的下一个比
                            if k == len(suppress) - 1:
                                if len(suppress) < max_hand_num:
                                    suppress.append(boxes[i])
                                # print "suppress3", suppress
                            else:
                                continue
        elif len(boxes) == 1:
            suppress = boxes

        # print "suppress", suppress
        return suppress


    def predict_camera(self, ort_sess_call, ort_sess_faceDetect, ort_sess_faceLandmark):
        """
        :param net:
        :param path:
        :return:
        """
        landmarks_right, landmarks_left, dst_w, dst_h = self.landmarks_call_train_bigRect()

        video_capture = cv2.VideoCapture(0)
        video_capture.set(3, 1280)  # 设置分辨率
        video_capture.set(4, 720)
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if ret:
                if frame is None:
                    exit()

                img_h = frame.shape[0]
                img_w = frame.shape[1]

                frame_copy = frame.copy()

                boxesOri = facedetect_onnx.predict_images_onnx(ort_sess_faceDetect, frame)
                if boxesOri == []:
                    continue

                faceRectsFinal = []
                for i, pred in enumerate(boxesOri):
                    if pred[1] >= 0.2:
                        # print('boxesOri', boxesOri)
                        rects = boxesOri[i][2:]
                        rects[0], rects[1], rects[2], rects[3] = rects[0] * img_w, rects[1] * img_h, rects[2] * img_w, rects[3] * img_h
                        faceW = rects[2] - rects[0]
                        faceH = rects[3] - rects[1]
                        faceCenter = [rects[0] + faceW / 2, rects[1] + faceH / 2]
                        offsetFace = int(max(faceW, faceH) / 2.)
                        faceStart = [max(0, int(faceCenter[0] - offsetFace)), max(0, int(faceCenter[1] - offsetFace))]
                        faceEnd = [min(frame.shape[1], int(faceCenter[0] + offsetFace)),
                                   min(frame.shape[0], int(faceCenter[1] + offsetFace))]
                        faceRectsFinal.append([faceStart[0], faceStart[1], faceEnd[0], faceEnd[1]])
                        cv2.rectangle(frame_copy, (faceStart[0], faceStart[1]), (faceEnd[0], faceEnd[1]), (0, 255, 0), 2)
                        fscore = pred[1]
                        fscore = round(fscore, 3)

                points, angle = facelandmark.main_image_onnx(ort_sess_faceLandmark, frame, faceRectsFinal[0])
                if points == []:
                    continue

                for x, y in points:
                    cv2.circle(frame_copy, (int(x), int(y)), 1, (255, 0, 0), 2)
                # cv2.putText(frame_copy, str(angle), (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                boundingbox_randomCut = [7, 10, 193, 149.5]
                SIZE = (64, 48) # w, h

                for directionIsLeft in range(2):
                    if directionIsLeft == 1:
                        # 人的左耳，主驾靠窗，（图片右侧的耳朵
                        cor_img, cor_pts = self.alignWithLandmark(frame, np.array(points), np.array(landmarks_left),
                                                                 dst_w, dst_h)
                    else:
                        # 人的右耳，不靠窗的一侧，（图片左侧的耳朵
                        cor_img, cor_pts = self.alignWithLandmark(frame, np.array(points), np.array(landmarks_right),
                                                                 dst_w, dst_h)

                    cor_img = Image.fromarray(cor_img)
                    img_crop = cor_img.crop(boundingbox_randomCut)
                    img_crop = np.array(img_crop)

                    img_crop = cv2.resize(img_crop, SIZE, interpolation=cv2.INTER_AREA)
                    img_net = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)

                    img_net = np.expand_dims(img_net, axis=0)
                    img_net = np.expand_dims(img_net, axis=0)
                    img_net = img_net.astype(np.float32)

                    img_net7 = {ort_sess_call.get_inputs()[0].name: img_net}
                    pred_class = ort_sess_call.run(None, img_net7)
                    prob = np.exp(pred_class) / np.exp(pred_class).sum()  # 归一化
                    prob = prob[0][0]
                    thePredict = prob.argsort()[-1]

                    if thePredict == 1 and prob[1] >= 0.6:
                        thePredict = 1
                    else:
                        thePredict = 0

                    print('thePredict, prob', thePredict, prob)
                    self.show(frame_copy, thePredict, directionIsLeft)
                    cv2.imshow('img_crop_'+str(directionIsLeft), img_crop)

                cv2.imshow('frame_copy', frame_copy)
                cv2.waitKey(0)
            else:
                break


if __name__ == '__main__':
    import onnxruntime
    # --------------call---------------------------
    Danger = DMSDanger()
    # paddle.device.set_device("gpu:0")

    onnx_file_faceDetect = "../../models/onnx_models/FaceDetection0114V4Main.onnx"
    onnx_facept = "../../models/onnx_models/FacialLandmark220324V25Main.onnx"
    onnx_call = '../../models/onnx_models/DMSCall220720V6MainQATFuseBnFuse1Layer3000_sim_skipbn.onnx'
    ort_sess_faceDetect = onnxruntime.InferenceSession(onnx_file_faceDetect)
    ort_sess_faceLandmark = onnxruntime.InferenceSession(onnx_facept)
    ort_sess_call = onnxruntime.InferenceSession(onnx_call)

    print('\n Load {} success! \n'.format(onnx_call))

    # camera
    Danger.predict_camera(ort_sess_call, ort_sess_faceDetect, ort_sess_faceLandmark)






