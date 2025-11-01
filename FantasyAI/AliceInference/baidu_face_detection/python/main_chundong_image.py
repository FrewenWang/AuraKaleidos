"""
使用模板匹配的人脸跟踪
使用嘴巴内嘴唇上下点-上下嘴唇的比例
可以调用摄像头观察效果
"""

import copy
import math
import onnxruntime
import cv2 as cv
import numpy as np
import template

template = template.Templates()


# 计算两个点的距离
def compute_pt_dist(pt1, pt2):
    x1, y1 = pt1
    x2, y2 = pt2

    dist1 = (x1 - x2) ** 2 + (y1 - y2) ** 2
    dist2 = math.sqrt(dist1)
    return dist2


# 获得人脸关键点
def get_facept_onnx(facept_Model, img_face, inputW, inputH):
    """
    :param facept_Model: 人脸关键点模型
    :param img_face:　人脸图片
    :param inputW:　模型输入的数据的宽度
    :param inputH: 模型输入的数据的高度
    :return:　返回人脸关键点相关信息
    """
    norm_w, norm_h = inputW, inputH
    pt_rslt = []
    data_draw = copy.deepcopy(img_face)
    data_out = cv.resize(img_face, (norm_w, norm_h))

    data = cv.cvtColor(data_out, cv.COLOR_BGR2GRAY)
    m, s = cv.meanStdDev(data)
    data2 = (data - m) / (1e-6 + s)

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
    # print (output_rslt1[0], "pitch-yaw-roll:", round(pitch,2), round(yaw,2), round(roll,2), output_rslt1[-1])
    data_out_h, data_out_w, _ = data_draw.shape
    for i in range(106):
        px = output_rslt1[2 * i + 1]
        py = output_rslt1[2 * i + 2]
        pt_rslt.append([px * data_out_w, py * data_out_h])

        # px = int(px * data_out_w)
        # py = int(py * data_out_h)
        # cv.circle(data_draw, (int(px*data_out_w),int(py*data_out_h)), 3, (0,0,255), -1, 8, 0)
    # cv.imshow("faceImg_draw", data_draw)
    # cv.imwrite("/media/baidu/ssd2/traindemo/facept_demo/" + str(id) + ".jpg", data_draw)
    # cv.waitKey(0)
    return [face_cls, pt_rslt, pitch, yaw, roll, eye_status]


# 人脸检测类
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
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image_origin = cv.resize(image, (288, 160))
        image_pre = (image_origin.astype(np.float32) / 255. - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        # print('resized image shape:', image_pre.shape)
        image = np.expand_dims(image_pre, axis=0)

        image = (image.astype(np.float32))
        image = np.transpose(image, [0, 3, 1, 2])

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


def main_facept_image():
    len_threshvalue = 10  # 滑窗的长度,可根据fps调整
    fratio_mouth_close = 0.2  # 闭嘴数据应占的比例
    # fratio_mouth_openSlight = 0.3 #小幅度张嘴数据应占的比例
    fratio_mouth_openBig = 0.2  # 大幅度张嘴数据应占的比例
    threshvalue_zhangbizui = 0.18  # 判断阈值，大于此阈值则认为张嘴，否则认为闭嘴

    # 人脸检测模型
    facedetect = FaceDetectOnnx(pathModel="../models/onnx_models/FaceDetection0412V4Main_quant.onnx")
    net_inputH = 128
    net_inputW = 128

    # 人脸关键点模型
    onnx_facept = "../models/onnx_models/FacialLandmark220419V25Main.onnx"
    ort_sess = onnxruntime.InferenceSession(onnx_facept)

    # 存放人脸的模板
    template_face_prefix = []  # 存放人脸匹配的模板

    # rslt_inUpDown_LR = []
    # rslt_outUpDown_LR = []
    # rslt_inUpDown_outUpDown = []
    # 内上，内下嘴唇点距离　除以　外上内上嘴唇点距离+外下内上嘴唇点距离
    rslt_inUpDown_lipUpDown = []

    # 如果是摄像头的话，此处修改为cap = cv.VideoCapture(0)即可
    # 如果摄像头是640x480的，可修改为1280x720
    # cap.set(3, 1280)
    # cap.set(4, 720)
    cap = cv.VideoCapture("../res/demo.mp4")

    while 1:
        ret, img = cap.read()
        if not ret:
            break
        if img is None:
            break
        img_draw = copy.deepcopy(img)
        img_forMouth = copy.deepcopy(img)
        imgH, imgW, _ = img.shape

        boxRslt = []
        if len(template_face_prefix) == 0:
            # 获取人脸检测方框
            boxRsltTmp = facedetect.getfacebox(image=img)
            if len(boxRsltTmp) > 0:
                boxRslt = [boxRsltTmp[0]]
        else:
            # demo设置只有一个人脸，具体使用时应参考实际情况
            # 实际使用时，使用模板匹配进行跟踪，跟踪方法与旧方法一致，工程无需更改
            faceRectTmp = template.useTemplate(img, template_face_prefix[0])
            if len(faceRectTmp) != 0:
                boxRslt.append([1.0] + faceRectTmp)
            template_face_prefix.clear()
        # 此处示例只使用一张人脸
        if len(boxRslt) > 1:
            boxRslt = [boxRslt[0]]
        for boxitem in boxRslt:
            # 分析人脸方框
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
            imgface = img[t:b, l:r, :]  # 获得人脸图片
            # 获得人脸关键点结果
            facecls_score, pt_rslt, fpitch, fyaw, froll, faceeye_score = get_facept_onnx(ort_sess, imgface, net_inputW,
                                                                                         net_inputH)
            if facecls_score < 0.15:
                template_face_prefix.clear()
                continue
            # 将人脸关键点映射到原图上
            pt_for_template = []
            for id in range(len(pt_rslt)):
                px, py = pt_rslt[id]
                px, py = int(px + l), int(py + t)
                pt_for_template.append([px, py])
                cv.circle(img_draw, (px, py), 2, (0, 255, 0), -1, 8, 0)
                if id % 2 == 0:
                    cv.putText(img_draw, str(id + 1), (px, py), 1, 1, (0, 0, 255), 1, 8, 0)
            # 制作模板
            template_face_tmp = template.createTemplate(img, pt_for_template)
            template_face_prefix.clear()
            # 　存储模板
            template_face_prefix.append(template_face_tmp)

            # 将人脸关键点模型的结果输出到图片上
            cv.rectangle(img_draw, (l, t), (r, b), (0, 255, 255), 3, 8, 0)
            fscore = round(score, 3)
            cv.putText(img_draw, str(fscore) + "|" + str(r - l) + "|" + str(b - t), (l, t), 1, 3, (0, 0, 255), 3, 8, 0)
            cv.putText(img_draw, "facecls:" + str(round(facecls_score, 2)), (50, 50), 1, 2, (0, 0, 255), 2, 8, 0)
            cv.putText(img_draw, "pitch:" + str(round(fpitch, 2)), (50, 90), 1, 2, (0, 0, 255), 2, 8, 0)
            cv.putText(img_draw, "yaw:" + str(round(fyaw, 2)), (50, 130), 1, 2, (0, 0, 255), 2, 8, 0)
            cv.putText(img_draw, "roll:" + str(round(froll, 2)), (50, 170), 1, 2, (0, 0, 255), 2, 8, 0)
            # >0.5 open,   <=0.5 close
            cv.putText(img_draw, "faceeye:" + str(round(faceeye_score, 2)), (50, 210), 1, 2, (0, 0, 255), 2, 8, 0)


            # 取嘴巴上的几个点进行分析，后续优化主要依赖这几个点
            pt_99 = pt_rslt[99]  # 内上嘴唇
            pt_102 = pt_rslt[102]  # 内下嘴唇
            pt_88 = pt_rslt[88]  # 外上嘴唇
            pt_91 = pt_rslt[91]  # 外下嘴唇
            # pt_96 = pt_rslt[96]   #内左嘴唇
            # pt_97 = pt_rslt[97]   #内右嘴唇

            # 画出嘴巴上的几个点
            cv.circle(img_forMouth, (int(pt_99[0] + l + 0.5), int(pt_99[1] + t + 0.5)), 2, (0, 255, 0), -1, 8, 0)
            cv.circle(img_forMouth, (int(pt_102[0] + l + 0.5), int(pt_102[1] + t + 0.5)), 2, (0, 255, 0), -1, 8, 0)
            cv.circle(img_forMouth, (int(pt_88[0] + l + 0.5), int(pt_88[1] + t + 0.5)), 2, (0, 255, 0), -1, 8, 0)
            cv.circle(img_forMouth, (int(pt_91[0] + l + 0.5), int(pt_91[1] + t + 0.5)), 2, (0, 255, 0), -1, 8, 0)
            # cv.circle(img_forMouth, (int(pt_96[0] + l + 0.5), int(pt_96[1] + t + 0.5)), 2, (0, 255, 0), -1, 8, 0)
            # cv.circle(img_forMouth, (int(pt_97[0] + l + 0.5), int(pt_97[1] + t + 0.5)), 2, (0, 255, 0), -1, 8, 0)

            dist_99_102 = compute_pt_dist(pt_99, pt_102)  # 内上嘴唇点和内下嘴唇点的距离
            dist_99_88 = compute_pt_dist(pt_99, pt_88)  # 内上嘴唇点和外上嘴唇点的距离
            dist_102_91 = compute_pt_dist(pt_102, pt_91)  # 内下嘴唇点和外下最准点的距离
            # dist_88_91 = compute_pt_dist(pt_88, pt_91)
            # dist_96_97 = compute_pt_dist(pt_96, pt_97)

            # cv.putText(img_forMouth, "left-right:"+str(round(dist_96_97, 3)), (50, 50), 1, 2, (0,0,255), 2, 8, 0)
            cv.putText(img_forMouth, "inUp-inDown:" + str(round(dist_99_102, 3)), (50, 90), 1, 2, (0, 0, 255), 2, 8, 0)
            # cv.putText(img_forMouth, "outUp-outDown" + str(round(dist_88_91, 3)), (50, 130), 1, 2, (0,0,255), 2, 8, 0)
            cv.putText(img_forMouth, "outUp-inUp" + str(round(dist_99_88, 3)), (50, 170), 1, 2, (0, 0, 255), 2, 8, 0)
            cv.putText(img_forMouth, "inDown-outDown" + str(round(dist_102_91, 3)), (50, 210), 1, 2, (0, 0, 255), 2, 8,
                       0)

            # fratio_inUpDown_LR = dist_99_102 / dist_96_97
            # fratio_outUpDown_LR = dist_88_91 / dist_96_97
            # fratio_inUpDown_outUpDown = dist_99_102 / dist_88_91
            fratio_inUpDown_lipUpDown = dist_99_102 / (dist_99_88 + dist_102_91)
            print("fratio_inUpDown_lipUpDown:", fratio_inUpDown_lipUpDown)
            # cv.putText(img_forMouth, "inUpDown-LR:"+str(round(fratio_inUpDown_LR,3)), (50, 300), 1, 2, (0,0,255), 2, 8, 0)
            # cv.putText(img_forMouth, "outUpDown-LR:"+str(round(fratio_outUpDown_LR, 3)), (50, 340), 1, 2, (0,0,255), 2, 8, 0)
            # cv.putText(img_forMouth, "inUpDown-outUpDown:"+str(round(fratio_inUpDown_outUpDown, 3)), (50, 380), 1, 2, (0,0,255), 2, 8, 0)
            cv.putText(img_forMouth, "inUpDown-lipsUpDown:" + str(round(fratio_inUpDown_lipUpDown, 3)), (50, 420), 1, 2,
                       (0, 0, 255), 2, 8, 0)

            # if len(rslt_inUpDown_LR) >= len_threshvalue:
            #     rslt_inUpDown_LR.pop(0)
            # rslt_inUpDown_LR.append(fratio_inUpDown_LR)
            # if len(rslt_outUpDown_LR) >= len_threshvalue:
            #     rslt_outUpDown_LR.pop(0)
            # rslt_outUpDown_LR.append(fratio_outUpDown_LR)
            # if len(rslt_inUpDown_outUpDown) >= len_threshvalue:
            #     rslt_inUpDown_outUpDown.pop(0)
            # rslt_inUpDown_outUpDown.append(fratio_inUpDown_outUpDown)
            # 将每一帧的结果保存到滑窗中
            if len(rslt_inUpDown_lipUpDown) >= len_threshvalue:
                rslt_inUpDown_lipUpDown.pop(0)
            rslt_inUpDown_lipUpDown.append(fratio_inUpDown_lipUpDown)

            # 分析内嘴唇的上下点占左右点的比例
            rslt_inUpDown_lipUpDown_array = np.array(rslt_inUpDown_lipUpDown)
            num_bizui = np.sum(rslt_inUpDown_lipUpDown_array < threshvalue_zhangbizui)  # 闭嘴的数量
            num_zhangzui = np.sum(rslt_inUpDown_lipUpDown_array >= threshvalue_zhangbizui)  # 张嘴的数量

            # 默认处于非唇动状态
            mouth_move_status = "not move"
            threashvalue_bizui = int(len_threshvalue * fratio_mouth_close)  # 闭嘴数据应占的比例
            threashvalue_zhangzui = int(len_threshvalue * fratio_mouth_openBig)  # 张嘴数据应占的比例
            # print (len(rslt_inUpDown_lipUpDown), "bizui:",num_bizui, "zhangzui:",num_zhangzui, threashvalue_bizui, threashvalue_zhangzui)
            # 如果闭嘴数量大于阈值，张嘴数据大于阈值则认为处于唇动状态
            if num_bizui > threashvalue_bizui and num_zhangzui > threashvalue_zhangzui:
                mouth_move_status = "move"

            # 将唇动结果显示在图片上
            if mouth_move_status == 'move':
                cv.putText(img_forMouth, "mouth:" + mouth_move_status, (500, 100), 1, 2, (0, 255, 255), 2, 8, 0)
            else:
                cv.putText(img_forMouth, "mouth:" + mouth_move_status, (500, 100), 1, 2, (255, 0, 0), 2, 8, 0)

            cv.imshow("img_mouth", img_forMouth)
            cv.imshow("img_draw", img_draw)
            cv.waitKey(1)


if __name__ == "__main__":
    main_facept_image()
