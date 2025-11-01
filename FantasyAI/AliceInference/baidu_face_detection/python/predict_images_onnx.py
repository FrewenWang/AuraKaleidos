#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
predict images
"""
import os
import cv2
import numpy as np
from PIL import Image
# from paddleslim.dygraph.quant import QAT
import onnxruntime
from scipy.linalg import solve

import facelandmark

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


class DMSDanger(object):

    def show(self, frame_copy, DMSType, useModel='useDMSClass7'):
        """
        0: normal  1:smoke  2:silence  3:drink  4: open  5:mask  6:捂嘴(挡嘴)  7.有手(不挡嘴)
        """
        x1, y0 = 200, 200
        # print('DMSType', DMSType)
        if useModel == 'use211124Model':
            if DMSType == 0:
                cv2.putText(frame_copy, "normal", (int(x1), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif DMSType == 1:
                cv2.putText(frame_copy, "open", (int(x1), int(y0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif DMSType == 2:
                cv2.putText(frame_copy, "smoke", (int(x1), int(y0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif DMSType == 3:
                cv2.putText(frame_copy, "silence", (int(x1), int(y0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif DMSType == 4:
                cv2.putText(frame_copy, "drink", (int(x1), int(y0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif DMSType == 5:
                cv2.putText(frame_copy, "mask", (int(x1), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif DMSType == 6:
                cv2.putText(frame_copy, "wuzui", (int(x1), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        elif useModel == 'useDMSClass7':
            if DMSType == 1:
                cv2.putText(frame_copy, "smoke", (int(x1), int(y0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif DMSType == 2:
                cv2.putText(frame_copy, "silence", (int(x1), int(y0)),  # silence &
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif DMSType == 3:
                cv2.putText(frame_copy, "drink", (int(x1), int(y0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif DMSType == 4:
                cv2.putText(frame_copy, "open", (int(x1), int(y0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # else:
            # cv2.putText(frame_copy, "normal", (int(x1), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            elif DMSType == 0:
                cv2.putText(frame_copy, "normal", (int(x1), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif DMSType == 5:
                cv2.putText(frame_copy, "mask", (int(x1), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif DMSType == 6:
                cv2.putText(frame_copy, "wuzui", (int(x1), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # elif DMSType == 7:
            # cv2.putText(frame_copy, "hand-other", (int(x1), int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            exit()

    def landmarks106_five(self, a, b, c):
        """xingqi--五分类"""
        landmark106 = np.array(landmark106_new)
        dela = np.array(
            [0, 1, 31, 32] + [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50] + [51, 52, 53, 54,
                                                                                                         57, 59, 60, 61,
                                                                                                         62, 63, 64, 67,
                                                                                                         69, 70, 75, 71,
                                                                                                         76])
        landmark106_ = np.delete(landmark106, np.array(dela), axis=0)

        face_w, face_h = landmark106_.max(axis=0) - landmark106_.min(axis=0)
        face_w = max([face_w, face_h])
        face_h = face_w

        start = landmark106_.min(axis=0)
        dx = face_w * a - start[0]
        dy = face_h * b - start[1]
        landmark106 = landmark106 + np.array([dx, dy])
        return landmark106, int(c * face_h), int(c * face_w)

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

    def alignWithLandmark(self, image, pts, landmark106, face_w_2, face_h_2):
        """crop"""
        M = self.align1(pts, landmark106)
        cor_img = cv2.warpAffine(image, M, (face_w_2, face_h_2))
        cor_pts = np.dot(M, np.column_stack((pts, [1] * pts.shape[0])).T).T
        cor_pts = cor_pts[:, :2]
        return cor_img, cor_pts

    def mat_inter(self, box1, box2):
        """判断两个矩形是否相交 box=(xA,yA,xB,yB)"""
        # print('box1, box2', box1, box2)

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

    def cover_face(self, points, faceH, frame, cover=False):
        xiaba = points[16]
        zuichun_xiazhong = points[91]

        # print('xiaba', xiaba)
        # print('zuichun_xiazhong', zuichun_xiazhong)
        # cv2.circle(frame_copy, (xiaba[0], xiaba[1]), 1, (255, 255, 0), 2)
        # cv2.circle(frame_copy, (zuichun_xiazhong[0], zuichun_xiazhong[1]), 1, (255, 255, 0), 2)

        if cover == '1_2':
            coverY = int(zuichun_xiazhong[1] + float(xiaba[1] - zuichun_xiazhong[1]) / float(2))  # xiaba 1/2
        elif cover == '1_8':
            coverY = int(xiaba[1] + (float(faceH) / 8.0))  # rect 1/8
        else:
            return frame

        maskH = max(0, int(frame.shape[0] - coverY))
        randomByteArray = bytearray(os.urandom(maskH * frame.shape[1] * 3))
        flatNumpyArray = np.array(randomByteArray)
        mask = flatNumpyArray.reshape(maskH, frame.shape[1], 3)
        if maskH > 0:
            frame[coverY:frame.shape[0], :, :] = mask
        return frame

    def predict_camera(self, ort_sess_dms, ort_sess_faceDetect, ort_sess_faceLandmark):
        """
        :param net:
        :param path:
        :return:
        """
        dst_landmarks, dst_h, dst_w = self.landmarks106_five(0.3, 0.3, 1.6)

        video_capture = cv2.VideoCapture(0)
        # video_capture.set(3, 1600)  # 设置分辨率
        # video_capture.set(4, 1300)
        video_capture.set(3, 1280)  # 设置分辨率
        video_capture.set(4, 720)
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if ret:
                if frame is None:
                    exit()

                img_h = frame.shape[0]
                img_w = frame.shape[1]
                N = frame.shape[2]

                frame_copy = frame.copy()

                boxesOri = facedetect_onnx.predict_images_onnx(ort_sess_faceDetect, frame)
                if boxesOri == []:
                    continue

                faceRectsFinal = []
                for i, pred in enumerate(boxesOri):
                    if pred[1] >= 0.2:
                        # print('boxesOri', boxesOri)
                        rects = boxesOri[i][2:]
                        rects[0], rects[1], rects[2], rects[3] = rects[0] * img_w, rects[1] * img_h, rects[2] * img_w, \
                                                                 rects[3] * img_h
                        faceW = rects[2] - rects[0]
                        faceH = rects[3] - rects[1]
                        faceCenter = [rects[0] + faceW / 2, rects[1] + faceH / 2]
                        offsetFace = int(max(faceW, faceH) / 2.)
                        faceStart = [max(0, int(faceCenter[0] - offsetFace)), max(0, int(faceCenter[1] - offsetFace))]
                        faceEnd = [min(frame.shape[1], int(faceCenter[0] + offsetFace)),
                                   min(frame.shape[0], int(faceCenter[1] + offsetFace))]
                        faceRectsFinal.append([faceStart[0], faceStart[1], faceEnd[0], faceEnd[1]])
                        cv2.rectangle(frame_copy, (faceStart[0], faceStart[1]), (faceEnd[0], faceEnd[1]), (0, 255, 0),
                                      2)
                        fscore = pred[1]
                        fscore = round(fscore, 3)

                points, angle = facelandmark.main_image_onnx(ort_sess_faceLandmark, frame, faceRectsFinal[0])
                if points == []:
                    continue

                for x, y in points:
                    cv2.circle(frame_copy, (int(x), int(y)), 1, (255, 0, 0), 2)
                cv2.putText(frame_copy, str(angle), (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                boundingbox_randomCut = [45, 45, 195, 195]
                SIZE = (112, 112)

                cor_img, cor_pts = self.alignWithLandmark(frame, np.array(points), np.array(dst_landmarks),
                                                          dst_w, dst_h)
                # print('cor_img, dst_h, dst_w, useModel', cor_img.shape, dst_h, dst_w, useModel)
                cor_img = Image.fromarray(cor_img)
                img_crop = cor_img.crop(boundingbox_randomCut)
                img_crop = np.array(img_crop)
                # print('img_crop1, SIZE', img_crop.shape, SIZE)

                img_crop = cv2.resize(img_crop, SIZE, interpolation=cv2.INTER_AREA)
                img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
                # print('img_crop2', img_crop.shape)

                m, s = cv2.meanStdDev(img_crop)
                img_net = (img_crop - m) / (1e-6 + s)
                img_net = np.expand_dims(img_net, axis=0)
                img_net = np.expand_dims(img_net, axis=0)
                img_net = (img_net.astype(np.float32))
                # img_net = paddle.to_tensor([[img_net]], dtype='float32')

                ort_inputs = {ort_sess_dms.get_inputs()[0].name: img_net}
                pred_class = ort_sess_dms.run(None, ort_inputs)
                pred_class = pred_class[0][0]
                prob = np.exp(pred_class) / np.exp(pred_class).sum()  # 归一化
                thePredict = prob.argsort()[-1]

                if thePredict == 1 and prob[1] < 0.7:
                    thePredict = 0

                print('thePredict', thePredict, prob)
                # cv2.imwrite(img_path.replace('.jpg', '_crop_thePredict{}_prob{}.jpg'.format(str(thePredict), str(prob))), img_crop)
                # img_net.tofile(img_path.replace('.jpg', '_crop_thePredict{}_prob{}.raw'.format(str(thePredict), str(prob))))
                # cv2.waitKey(15)

                self.show(frame_copy, thePredict)
                cv2.putText(frame_copy, str(prob), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.imshow('frame_copy', frame_copy)
                cv2.imshow('img_crop', img_crop)
                # cv2.imwrite('./log_test/' + str(random.randint(1,100))+'.jpg', frame)
                cv2.waitKey(0)
            else:
                exit()


if __name__ == '__main__':
    # paddle.device.set_device("gpu:1")
    onnx_file_faceDetect = "../onnx_model/FaceDetection0412V4Main_quant.onnx"
    onnx_facept = "../onnx_model/FacialLandmark220419V25Main.onnx"
    onnx_dms = '../onnx_model/DMSSenven220422V11MainQAT.onnx'
    ort_sess_faceDetect = onnxruntime.InferenceSession(onnx_file_faceDetect)
    ort_sess_faceLandmark = onnxruntime.InferenceSession(onnx_facept)
    ort_sess_dms = onnxruntime.InferenceSession(onnx_dms)

    # --------------DMS----------------------------
    Danger = DMSDanger()
    print('\n Load {} success! \n'.format(onnx_dms))

    # camera
    Danger.predict_camera(ort_sess_dms, ort_sess_faceDetect, ort_sess_faceLandmark)
