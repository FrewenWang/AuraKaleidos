#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
predict images
"""
import paddle.distributed as dist
import paddle
import os
import shutil
import sys
import time
import copy
import json
import cv2
import numpy as np
import random
import math
from PIL import Image
from scipy.linalg import solve
from imgaug import augmenters as iaa
from paddleslim.dygraph import L1NormFilterPruner
from paddleslim.dygraph.quant import QAT
import onnxruntime
from collections import OrderedDict
import multiprocessing as mp

# import write_html
sys.path.append('../')

# import facedetect_onnx
# import main_facept_onnx as facelandmark
import main_facept_onnx_1025 as facelandmark
import facedetect_onnx_1107 as facedetect_onnx


quant_config = {
    'weight_preprocess_type': None,
    'activation_preprocess_type': None,
    'weight_quantize_type': 'abs_max',
    'activation_quantize_type': 'moving_average_abs_max',
    'weight_bits': 8,
    'activation_bits': 8,
    'dtype': 'int8',
    'window_size': 10000,
    'moving_rate': 0.9,
    'quantizable_layer_type': ['Conv2D', 'Linear'],
}

landmark106 = [(2, 27),  # leftface
               (3, 36),
               (4, 44),
               (6, 53),
               (8, 62),
               (11, 70),
               (14, 78),
               (17, 84),
               (21, 90),
               (25, 94),
               (29, 99),
               (34, 102),
               (38, 105),
               (42, 107),
               (46, 109),
               (52, 110),
               (56, 111),  # jaw
               (109, 27),  # rightface
               (108, 36),
               (107, 44),
               (105, 53),
               (103, 62),
               (100, 70),
               (97, 78),
               (94, 84),
               (90, 90),
               (86, 94),
               (82, 99),
               (77, 102),
               (73, 105),
               (69, 107),
               (65, 109),
               (59, 110),
               (7, 13),  # lefteyebrow
               (14, 7),
               (23, 6),
               (33, 8),
               (42, 12),
               (15, 14),
               (24, 14),
               (33, 14),
               (42, 14),
               (104, 13),  # righteyebrow
               (97, 7),
               (88, 6),
               (78, 8),
               (69, 12),
               (96, 14),
               (87, 14),
               (78, 14),
               (69, 14),
               (18, 27),  # lefteye
               (24, 25),
               (31, 25),
               (38, 29),
               (31, 30),
               (24, 30),
               (28, 25),
               (27, 30),
               (28, 27),
               (28, 28),
               (93, 27),  # righteye
               (87, 25),
               (80, 25),
               (73, 29),
               (80, 30),
               (87, 30),
               (83, 25),
               (84, 30),
               (83, 27),
               (83, 28),
               (56, 26),  # nose
               (56, 38),
               (56, 50),
               (56, 61),
               (48, 29),
               (64, 28),
               (43, 54),
               (69, 53),
               (37, 59),
               (75, 59),
               (45, 64),
               (49, 67),
               (56, 69),
               (63, 67),
               (67, 64),
               (38, 85),  # mouth
               (51, 87),
               (56, 86),
               (61, 87),
               (74, 85),
               (56, 95),
               (44, 86),
               (68, 86),
               (68, 90),
               (44, 90),
               (39, 85),
               (73, 85),
               (49, 84),
               (56, 84),
               (63, 84),
               (49, 87),
               (56, 88),
               (63, 87),
               (52, 94),
               (60, 94), ]

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

FACIAL_LANDMARKS_106_IDXS = OrderedDict([
    ("mouth", (86, 105)),
    ("right_eyebrow", (33, 41)),
    ("left_eyebrow", (42, 50)),
    ("right_eye", (51, 60)),
    ("left_eye", (61, 70)),
    ("nose", (71, 85)),
    ("leftface", (0, 2)),
    ("rightface", (31, 33)),
    ("jaw", (16, 18))
])


class Face:
    """
    Face

    """

    def __init__(self):
        """

        """
        self.name = None
        self.oldname = None
        self.rectangle = None
        self.image = None
        self.container_image = None
        self.embedding = None


class FaceClass:
    """
    FaceClass

    """
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        """

        :param name:
        :param image_paths:
        """
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        """

        :return:
        """
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        """

        :return:
        """
        return len(self.image_paths)


class FaceIDClass():
    """
    FaceIDClass()

    """

    def __init__(self, name, emb, emb_mean):
        """

        :param name:
        :param emb:
        :param emb_mean:
        """
        self.name = name
        self.emb = emb
        self.emb_mean = emb_mean

    def __str__(self):
        """

        :return:
        """
        return self.name + ', ' + str(len(self.emb)) + ' images'

    def __len__(self):
        """

        :return:
        """
        return len(self.emb)


def load_pretrain_weight(model, weight):
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
                # logger.info(
                #    '{} not used, shape {} unmatched with {} in model.'.format(
                #        name, weight.shape, list(model_dict[name].shape)))
                ignore_weights.add(name)
        else:
            #logger.info('Redundant weight {} and ignore it.'.format(name))
            ignore_weights.add(name)

    for weight in ignore_weights:
        param_state_dict.pop(weight, None)

    model.set_dict(param_state_dict) 
    #logger.info('Finish loading model weights: {}'.format(weights_path))
    print('Finish loading model weights: {}'.format(weights_path))


def slim_sens(net, senFile, threshold):
    """
    # 2. 剪裁
    # 加载敏感度文件
    :param net:
    :param senFile:
    :param threshold:
    :return:
    """
    pruner = L1NormFilterPruner(net, [1, 3, 112, 112], sen_file=senFile)
    # print('pruner.sensitive:', pruner.sensitive())
    plan = pruner.sensitive_prune(threshold)  #, skip_vars=["conv2d_57.w_0"]

    # print("!FLOPs after pruning: ", paddle.flops(net, (1, 3, 64, 64))/1024.0/1024.0)
    # print("!Pruned FLOPs: %", round(plan.pruned_flops*100, 2))
    # print('threshold: ', threshold)
    return net


def print_with_time(str_in):
    """

    :param str_in:
    :return:
    """
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    str_with_time = time_str + ': ' + str_in
    print(str_with_time)


def get_image_paths(facedir):
    """

    :param facedir:
    :return:
    """
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_temp = []
        for img in images:
            if img[0] == '.':
                continue
            else:
                if img.endswith('jpg') or img.endswith(
                        'png') or img.endswith('bmp') or img.endswith('jpeg'):
                    image_temp.append(img)

        # image_temp.sort(key=lambda x: int(x[:-4]))
        for img in image_temp:
            if img[0] == '.':
                continue
            else:
                if img.endswith('jpg') or img.endswith(
                        'png') or img.endswith('bmp') or img.endswith('jpeg'):
                    image_paths.append(os.path.join(facedir, img))

    return image_paths


def get_dataset(path, has_class_directories=True):
    """

    :param path:
    :param has_class_directories:
    :return:
    """
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = sorted([path for path in os.listdir(path_exp)
                      if os.path.isdir(os.path.join(path_exp, path))])
    nrof_classes = len(classes)  # ID num
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(FaceClass(class_name, image_paths))
    print('!!dataset', len(dataset))
    return dataset


def align(srcimage, srcpts, pts):
    """

    :param srcimage:
    :param srcpts:
    :param pts:
    :return:
    """
    # convert the landmark (x, y)-coordinates to a NumPy array
    # dstpts = shapetonp(pts)
    dstpts = np.array(pts)
    # dstpts = pts
    # print nppts
    (dstlStart, dstlEnd) = FACIAL_LANDMARKS_106_IDXS["left_eye"]
    (dstrStart, dstrEnd) = FACIAL_LANDMARKS_106_IDXS["right_eye"]
    dstleftEyePts = dstpts[dstlStart:dstlEnd]
    dstrightEyePts = dstpts[dstrStart:dstrEnd]
    # compute the center of mass for each eye
    dstleftEyeCenter = dstleftEyePts.mean(axis=0).astype("int") + [2, 0]
    dstrightEyeCenter = dstrightEyePts.mean(axis=0).astype("int") - [2, 0]

    # srccpts = shapetonp(srcpts)
    # print nppts
    (srclStart, srclEnd) = FACIAL_LANDMARKS_106_IDXS["left_eye"]
    (srcrStart, srcrEnd) = FACIAL_LANDMARKS_106_IDXS["right_eye"]
    srcleftEyePts = srcpts[srclStart:srclEnd]
    srcrightEyePts = srcpts[srcrStart:srcrEnd]
    # compute the center of mass for each eye
    srcleftEyeCenter = srcleftEyePts.mean(axis=0).astype("int")
    srcrightEyeCenter = srcrightEyePts.mean(axis=0).astype("int")

    # compute the angle between the eye centroids
    dstdY = dstrightEyeCenter[1] - dstleftEyeCenter[1]
    dstdX = dstrightEyeCenter[0] - dstleftEyeCenter[0]
    dstangle = np.degrees(np.arctan2(dstdY, dstdX)) - 180

    srcdY = srcrightEyeCenter[1] - srcleftEyeCenter[1]
    srcdX = srcrightEyeCenter[0] - srcleftEyeCenter[0]
    srcangle = np.degrees(np.arctan2(srcdY, srcdX)) - 180

    # grab the rotation matrix for rotating and scaling the face
    dst_gravity = dstpts.mean(axis=0).astype("float")
    src_gravity = srcpts.mean(axis=0).astype("float")
    scale = (np.sum((dstpts - dst_gravity) ** 2) ** 0.5 / len(dstpts)) / (
        np.sum((srcpts - src_gravity) ** 2) ** 0.5 / len(srcpts))

    M = cv2.getRotationMatrix2D((0, 0), srcangle, scale)
    M[0, 2] = dst_gravity[0] - M[0, 0] * \
        src_gravity[0] - M[0, 1] * src_gravity[1]
    M[1, 2] = dst_gravity[1] - M[1, 0] * \
        src_gravity[0] - M[1, 1] * src_gravity[1]
    output = cv2.warpAffine(srcimage, M, (112, 112))
    return output


def facealign(img, net_detect, net_landmark):
    """

    :param img:
    :param net_detect:
    :param net_landmark:
    :return:
    """
    faces = []
    img_h = img.shape[0]
    img_w = img.shape[1]

    faceRectsFinal = facedetect_onnx.predict_image_onnx(net_detect, copy.deepcopy(img))

    if len(faceRectsFinal) > 0:
        for i, rect in enumerate(faceRectsFinal):
            face = Face()
            # cv2.rectangle(img, (rect[0], rect[1]), (rect[2],rect[3]), (0, 255, 0), 2)
            #pts = np.zeros((106, 2), dtype="int")

            # pts, angle = facelandmark.main_image_onnx(net_landmark, copy.deepcopy(img), rect)
            pts, angle = facelandmark.main_facept_img(net_landmark, copy.deepcopy(img), rect)
            pts = np.array(pts)

            up = np.min(pts[:, 1])
            down = np.max(pts[:, 1])
            left = np.min(pts[:, 0])
            right = np.max(pts[:, 0])

            mag_param = 0.05
            y_edge = int((down - up) * mag_param)
            x_edge = int((right - left) * mag_param)
            up = max(0, (up - y_edge))
            down = min(img.shape[0], (down + y_edge))
            left = max(0, (left - x_edge))
            right = min(img.shape[1], (right + x_edge))

            rect = []
            rect.append(left)
            rect.append(up)
            rect.append(right)
            rect.append(down)

            # for inde, point in enumerate(pts):
            #     cv2.circle(img, (point[0], point[1]), 1, (0, 0, 255), 2)
            # cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0))
            # cv2.imshow("2", img)
            # cv2.waitKey(0)

            # if haveFace == 1:
            face.image = align(img, pts, landmark106)
            faces.append(face)
            # face.image = crop(image, pts, M)
            # cv2.imshow('fuck111', face.image)
            # cv2.waitKey(0)

    return faces


def cropface(img, net_detect, net_landmark):
    """

    :param img:
    :param net_detect:
    :param net_landmark:
    :return:
    """
    faceAligned = facealign(img, net_detect, net_landmark)
    if faceAligned is not None:
        for face in faceAligned:
            # if face.image.size > 0:
            return face.image
            # else:
            #     return None
    else:
        return None


def processs_meanstd(img):
    """

    :param img:
    :return:
    """
    m, s = cv2.meanStdDev(img)
    b, g, r = cv2.split(img)
    B = (b - m[0]) / (s[0] + 1e-6)
    G = (g - m[1]) / (s[1] + 1e-6)
    R = (r - m[2]) / (s[2] + 1e-6)
    image = cv2.merge([B, G, R])
    return image


def processImage(img):
    """

    :param img:
    :return:
    """
    m = 127.5
    s = 128.0
    b, g, r = cv2.split(img)
    B = (b - m) / (s + 1e-6)
    G = (g - m) / (s + 1e-6)
    R = (r - m) / (s + 1e-6)
    image = cv2.merge([B, G, R])
    return image


def get_embeddings(
        dataset,
        onnx_file_faceDetect,
        onnx_facept,
        faceid_model_backbone,
        faceid_model_path,
        q_out,
        q_data_used, use_qat, use_onnx):
    """

    :param dataset:
    :param q_out:
    :param q_data_used:
    :return:
    """
    detect_net = onnxruntime.InferenceSession(onnx_file_faceDetect)
    landmark_net = onnxruntime.InferenceSession(onnx_facept)
    if use_onnx:
        facenet = onnxruntime.InferenceSession(faceid_model_path)
    else:
        if faceid_model_backbone == '1012V1_5':
            from backbone.FaceRecognize1012V1_5Main.model_hardSwish import  mxnet_mdoel as Model
            facenet = Model(Eval=True)
            
        # elif faceid_model_backbone == '1012V1':
        #     from backbone.FaceRecognize1012V1Main.FaceRecognize1012V1Main_paddlenet.x2paddle_code_V1_scale_False import mxnet_mdoel as Model
        #     facenet = Model(Eval=True)
        #
        # elif faceid_model_backbone == 'd5e2a453':
        #     from backbone.d5e2a453.resnet101basic_backbone import ResNetDec as Model
        #     facenet = Model(arch='158144313233323332222221222323232311131112121312121212111313100000000000000000000002121213100', Eval=True)

        print('[Flops M]',paddle.flops(facenet,(1,3,112,112)) /1000.0 /1000.0)

        #2. qat use_qat
        if use_prun:
            # 1. prun
            senFile = './SEN_V1_0815.pickle'
            facenet = slim_sens(facenet, senFile, sen_threshold)
            print("!FLOPs after pruning: {}, sen_threshold: {}".format(paddle.flops(facenet, (1, 3, 112, 112))/1000.0/1000.0, sen_threshold))

        if use_qat:
            quanter = QAT(config=quant_config)
            quanter.quantize(facenet)
            print('量化训练载入模型成功！')
            print('[Flops M]',paddle.flops(facenet,(1,3,112,112)) /1000.0 /1000.0)

        # faceid_model_path = './params2jit/quantInferenceModel_1/quant_0615_qat'
        #faceid_model_path = '/media/baidu/ssd4T1/paddle_faceid/params2jit/jitDMSmodel_1/jit_0615_qat'
        #facenet = paddle.jit.load(faceid_model_patkh)

        param_state_dict = paddle.load(faceid_model_path + '.pdparams')
        facenet.set_dict(param_state_dict) #
        print('\n Load {} success! \n'.format(faceid_model_path))
        facenet.eval()

        params2jit = False  # 动转静
        if params2jit:
            # paddle.jit.to_static(facenet, input_spec=[paddle.static.InputSpec(shape=[None, 3, 112, 112], dtype='float32')])
            # paddle.jit.save(facenet, "./params2jit/jitDMSmodel_None/jit_"+faceid_model_path.split('/')[-2]+'/'+faceid_model_path.split('/')[-1].replace('.pdparams', ''))
            paddle.jit.to_static(facenet, input_spec=[paddle.static.InputSpec(shape=[1, 3, 112, 112], dtype='float32')])
            paddle.jit.save(facenet, "./params2jit/jitDMSmodel_1/jit_"+faceid_model_path.split('/')[-2]+'/'+faceid_model_path.split('/')[-1].replace('.pdparams', ''))
            # print('\n Save jit success! \n')
            if use_qat:
                #paddle.jit.to_static(facenet, input_spec=[paddle.static.InputSpec(shape=[None, 3, 112, 112], dtype='float32')])
                #quanter.save_quantized_model(facenet, './params2jit/quantInferenceModel_None/quant_'+faceid_model_path.split('/')[-2]+'/'+faceid_model_path.split('/')[-1].replace('.pdparams', ''))
                paddle.jit.to_static(facenet, input_spec=[paddle.static.InputSpec(shape=[1, 3, 112, 112], dtype='float32')])
                quanter.save_quantized_model(facenet, './params2jit/quantInferenceModel_paddle2.3.1/quant_'+faceid_model_path.split('/')[-2]+'/'+faceid_model_path.split('/')[-1].replace('.pdparams', ''))
                print('\n Save QuantizedModel success! \n')


    faceid_feats = {}
    count = 0
    data_used = {}
    total = len(dataset)
    for cls in dataset:  # one person
        img_list = []
        emb = []
        file_used_list = []
        for image_path in cls.image_paths:
            print("---------------------------------------")
            print(image_path)
            img = cv2.imread(image_path)
            # RGB图像转化灰度图片
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            B = img_gray.copy()
            G = img_gray.copy()
            R = img_gray.copy()
            img = cv2.merge([B, G, R])
            print(img)
            if use_crop == 1:
                img = cropface(img, detect_net, landmark_net)
                # cv2.imshow('cropface img', img)

            if img is not None:
                img = cv2.resize(img, (112, 112))

                # save_path = image_path.replace('Ori', 'Crop')
                # if not os.path.exists(save_path.replace(save_path.split('/')[-1], '')):
                #     os.makedirs(save_path.replace(save_path.split('/')[-1], ''))
                # cv2.imwrite(save_path, img)  #save

                img = processs_meanstd(img)  # -m  /s

                img = img.transpose(2, 0, 1)  # 3*112*112
                img_list.append(img)  # paddle.to_tensor(img, dtype='float32')
                file_used_list.append(image_path)

                if use_onnx:
                    img_net = img.astype(np.float32)
                    img_net = np.expand_dims(img_net, axis=0)
                    img_net_input = {facenet.get_inputs()[0].name: img_net}
                    embeddings = facenet.run(None, img_net_input)[0][0]  # 1024
                    emb.append(embeddings / (np.linalg.norm(embeddings) + 1e-06))

                    # save_path = image_path.replace('Ori', 'Crop')
                    # if not os.path.exists(save_path.replace(save_path.split('/')[-1], '')):
                    #     os.makedirs(save_path.replace(save_path.split('/')[-1], ''))
                    # img_net.tofile(save_path.replace('.jpg', '_crop.raw'))  # save
            else:
                print('no face: %s' % image_path)
                continue

        if not use_onnx:
            # batch forward
            n = len(img_list)
            if len(img_list) > 0:
                images = np.stack(img_list)  # transfer to ndarray type
            else:
                continue
            batch_num = 0

            for i in range(int(n / batch_size)):
                images_batch = images[batch_size * i: batch_size * (i + 1)]
                images_batch = paddle.to_tensor(images_batch, dtype='float32')
                embeddings = facenet(images_batch)  # (n, 1024)
                for j in range(batch_size):
                    emb.append(embeddings[j] / (np.linalg.norm(embeddings[j]) + 1e-06))
                batch_num += 1

            last_batch_num = n % batch_size
            if last_batch_num != 0:
                images_batch = images[batch_size * batch_num:]
                images_batch = paddle.to_tensor(images_batch, dtype='float32')
                embeddings = facenet(images_batch)
                for j in range(last_batch_num):
                    emb.append(embeddings[j] / (np.linalg.norm(embeddings[j]) + 1e-06))

        emb_of_one_cls = np.array(emb)
        emb_of_one_cls_mean = np.mean(emb_of_one_cls, axis=0)
        faceid_feats[cls.name] = FaceIDClass(cls.name, emb_of_one_cls, emb_of_one_cls_mean)

        count += 1
        if count % 10 == 0:
            print_with_time('%d/%d, got feature!' % (count, total))
        data_used[cls.name] = FaceClass(cls.name, file_used_list)
    print('len of faceIDresultss', len(faceid_feats))
    q_out.put(faceid_feats)
    q_data_used.put(data_used)


def compare(
        dataset,
        dataset1,
        SamePeopleList,
        DiffPeopleList,
        txt_path,
        compare_type,
        log_name,
        count_Pos,
        count_Neg,
        same_mean,
        diff_mean):
    """

    :param dataset:
    :param dataset1:
    :param SamePeopleList:
    :param DiffPeopleList:
    :param txt_path:
    :param compare_type:
    :param faceid_weight_file:
    :param count_Pos:
    :param count_Neg:
    :param same_mean:
    :param diff_mean:
    :return:
    """
    logname = txt_path + '/' + \
        log_name + '_' + compare_type + '.txt'

    fout = open(logname, 'w')
    fout.write('start...' + '\n')
    fout.flush()
    print_with_time('%s, start sort distance...' % compare_type)
    SamePeopleList.sort(key=lambda x: x[4])
    DiffPeopleList.sort(key=lambda x: -x[4])
    print_with_time('%s, sort over!' % compare_type)

    SameDist_np = np.array([item[4]
                           for item in SamePeopleList], dtype='double')
    DiffDist_np = np.array([item[4]
                           for item in DiffPeopleList], dtype='double')

    DiffMean = diff_mean / count_Neg
    SameMean = same_mean / count_Pos
    Diff_Min = DiffDist_np[-1]
    Same_Max = SameDist_np[-1]

    print_with_time('start writing txt')
    out = "Diff_Min %f Same_Max %f DiffMean %f SamePeopleMean %f DeltaMean %f" % (
        Diff_Min, Same_Max, DiffMean, SameMean, DiffMean - SameMean)  # print(out)
    fout.write(out + '\n')
    fout.flush()

    # k, 10k, 100k, 1m, 10m
    flag_err = [0 for _ in range(5)]  # if error rate arrived
    delta_err = [1e-07 for _ in range(5)]  # the order's error rate
    paras_err = ['' for _ in range(5)]  # output

    error_diff_10w = []
    error_diff_100w = []

    NegNum = count_Neg
    comp_num = int((thresh_range[1] - thresh_range[0]) * 200)
    Pos_ind = 0

    for i in range(comp_num):
        threshold = i * 0.005 + thresh_range[0]

        while Pos_ind < len(SameDist_np):
            if SameDist_np[Pos_ind] > threshold:
                break
            Pos_ind += 1
        falseNum = count_Neg - NegNum
        while falseNum < len(DiffDist_np):
            if DiffDist_np[-falseNum - 1] > threshold:  # reverse order
                break
            falseNum += 1

        NegNum = count_Neg - falseNum  # diff people not pass times
        # count_Neg: diff people compare times
        # falseNum: diff people dist < 0.35 + i*0.005

        # same people dist < 0.35 + i*0.005
        PosNum = Pos_ind + count_Pos - len(SameDist_np)
        # Pos_ind: same people dist > 0.35 and same people dist < 0.35 + i*0.005
        # count_Pos: same people compare times
        # len(SameDist_np):same people dist > 0.35

        passrate = float(PosNum) / count_Pos * 100.0  # 通过率
        errorrate = (1 - float(NegNum) / count_Neg) * 100.0  # 误检率
        out = "passrate-errorrate-threshold: %.4f, %f(%.3f) PosNum %d PosCount %d FasleNegNum %d NegCount %d" % (
            passrate, errorrate, threshold, PosNum, count_Pos, falseNum, count_Neg)
        fout.write(out + '\n')
        fout.flush()

        for k in range(5):
            delta_cur = errorrate - float('1e-0%d' % (k + 1))
            if flag_err[k] == 0:
                delta_err[k] = delta_cur
            if (flag_err[k] == 0 and delta_err[k] >= 0) or (
                    flag_err[k] == 1 and (delta_cur - delta_err[k]) < (delta_err[k] * 1e-05)):
                paras_err[k] = out
                flag_err[k] = 1
                if k == 2:
                    error_diff_10w = DiffPeopleList[-falseNum:]
                elif k == 3:
                    error_diff_100w = DiffPeopleList[-falseNum:]
    fout.write('\n' + "************************************")
    fout.write("key_params" + '\n')
    for k in range(5):
        fout.write(paras_err[-k - 1] + '\n')
    fout.close()

    file_error_diff10w = txt_path + '/' + \
        log_name + '_' + compare_type + '10wErr.txt'
    with open(file_error_diff10w, 'w') as fout_error:
        for info in error_diff_10w:
            image_path_a = dataset[info[0]].image_paths[info[1]]
            image_path_b = dataset1[info[2]].image_paths[info[3]]
            out = [image_path_a, image_path_b, str(info[4])]
            fout_error.write(','.join(out) + '\n')

    out_file = file_error_diff10w.replace('.txt', '.html')
    # write_html.csv_to_html(file_error_diff10w, out_file)

    # Same_35_50 = SamePeopleList[PosNum39:PosNum44]
    # file_Same3550 = txt_path + '/' + log_name.split('.')[0] + '_' + compare_type + 'Same_3944.txt'
    # with open(file_Same3550, 'w') as fout_error:
    #    for info in Same_35_50:
    #        image_path_a = dataset_a[info[0]].image_paths[info[1]]
    #        image_path_b = dataset_b[info[2]].image_paths[info[3]]
    #        out = [image_path_a, image_path_b, str(info[4])]
    #        fout_error.write(','.join(out) + '\n')
    # out_file = file_Same3550.replace('.txt', '.html')
    # write_html.csv_to_html(file_Same3550, out_file)

    same2000 = SamePeopleList[-2000:]
    file_same2000 = txt_path + '/' + \
        log_name + '_' + compare_type + 'Same2000.txt'
    with open(file_same2000, 'w') as fout_error:
        for info in same2000:
            image_path_a = dataset[info[0]].image_paths[info[1]]
            image_path_b = dataset1[info[2]].image_paths[info[3]]
            out = [image_path_a, image_path_b, str(info[4])]
            fout_error.write(','.join(out) + '\n')

    out_file = file_same2000.replace('.txt', '.html')
    # write_html.csv_to_html(file_same2000, out_file)
    print_with_time('%s, over!' % compare_type)


def compare_vecs(dataset, faceid_feats, txt_path, compare_type, log_name):
    """
    人脸特征的Vector的特征向量比对
    :param dataset:
    :param faceid_feats:
    :param txt_path:
    :param compare_type:
    :param faceid_weight_file:
    :return:
    """
    SamePeopleList = []
    DiffPeopleList = []
    count_Pos = 0
    count_Neg = 0
    same_mean = 0
    diff_mean = 0

    for i in range(len(faceid_feats.keys())):
        cls = list(faceid_feats.keys())[i]  # person1
        for j in range(len(faceid_feats[cls].emb)):
            vector_a = np.array(faceid_feats[cls].emb[j, :])  # img1
            for s in range(j + 1, len(faceid_feats[cls].emb)):
                vector_b = np.array(faceid_feats[cls].emb[s, :])  # img2
                dist = 1 - np.dot(vector_a, vector_b)
                count_Pos += 1
                same_mean += dist
                if dist >= thresh_range[0]:
                    # person i, image_rgb j, person s(vec_mean).
                    SamePeopleList.append([cls, j, cls, s, dist])
            for k in range(i + 1, len(faceid_feats.keys())):
                cls1 = list(faceid_feats.keys())[k]  # person2
                for d in range(0, len(faceid_feats[cls1].emb)):
                    vector_b = np.array(faceid_feats[cls1].emb[d, :])  # img2
                    dist = 1 - np.dot(vector_a, vector_b)
                    count_Neg += 1
                    diff_mean += dist
                    if dist <= thresh_range[1]:
                        # person cls, image i, person cls1, image j.
                        DiffPeopleList.append([cls, j, cls1, d, dist])
    print('SamePeopleList', len(SamePeopleList))
    print('DiffPeopleList', len(DiffPeopleList))
    compare(
        dataset,
        dataset,
        SamePeopleList,
        DiffPeopleList,
        txt_path,
        compare_type,
        log_name,
        count_Pos,
        count_Neg,
        same_mean,
        diff_mean)


def compare_vecs_twosets(
        dataset,
        faceid_feats,
        dataset1,
        faceid_feats1,
        txt_path,
        compare_type,
        log_name):
    """
    comapre dist of each pair of images between two datasets
    :param dataset:
    :param faceid_feats:
    :param txt_path:
    :param compare_type:
    :param faceid_weight_file:
    :return:
    """
    SamePeopleList = []
    DiffPeopleList = []
    count_Pos = 0
    count_Neg = 0
    same_mean = 0
    diff_mean = 0
    for i in range(len(faceid_feats.keys())):
        cls = list(faceid_feats.keys())[i]

        for j in range(len(faceid_feats[cls].emb)):
            vector_a = np.array(faceid_feats[cls].emb[j, :])
            for k in range(len(faceid_feats1.keys())):
                cls1 = list(faceid_feats1.keys())[k]
                for d in range(len(faceid_feats1[cls1].emb)):
                    vector_b = np.array(faceid_feats1[cls1].emb[d, :])
                    dist = 1 - np.dot(vector_a, vector_b)
                    print("======================================")
                    print(vector_a)
                    print(vector_b)
                    print(cls, cls1, dist)
                    if cls == cls1:
                        count_Pos += 1
                        same_mean += dist
                        if dist >= thresh_range[0]:
                            # person cls, image j, person cls, image d.
                            SamePeopleList.append([cls, j, cls, d, dist])
                    else:
                        count_Neg += 1
                        diff_mean += dist
                        if dist <= thresh_range[1]:
                            # person cls, image j, person cls1, image d.
                            DiffPeopleList.append([cls, j, cls1, d, dist])
    compare(
        dataset,
        dataset1,
        SamePeopleList,
        DiffPeopleList,
        txt_path,
        compare_type,
        log_name,
        count_Pos,
        count_Neg,
        same_mean,
        diff_mean)


class FaceID(object):  # GetValidDatasets
    """
    FaceID
    """

    def landmarks106_live(self, a, b, c):
        """
        live
        """
        landmark106 = np.array(landmark106_new)
        start = (landmark106[[0, 1, 2], 0].min(),
                 landmark106[[34, 35, 36, 43, 44, 45], 1].min())

        face_h = landmark106[16, 1] - \
            landmark106[[34, 35, 36, 43, 44, 45], 1].min()
        face_w = landmark106[[32, 31, 30], 0].max(
        ) - landmark106[[0, 1, 2], 0].min()
        face_h = face_w

        dx = face_w * a - start[0]
        dy = face_h * b - start[1]
        landmark106 = landmark106 + np.array([dx, dy])
        return landmark106, int(c * face_h), int(c * face_w)

    def landmarks_call_train_bigRect(self):
        """
        call
        :return:
        """
        cor_landmark106 = np.array(landmark106_new)
        face_w, face_h = cor_landmark106.max(
            axis=0) - cor_landmark106.min(axis=0)
        x_left = (cor_landmark106[52][0] + cor_landmark106[53]
                  [0] + cor_landmark106[58][0] + cor_landmark106[56][0]) / 4
        y_left = (cor_landmark106[52][1] + cor_landmark106[53]
                  [1] + cor_landmark106[58][1] + cor_landmark106[56][1]) / 4
        x_right = (cor_landmark106[62][0] + cor_landmark106[63]
                   [0] + cor_landmark106[68][0] + cor_landmark106[66][0]) / 4
        y_right = (cor_landmark106[62][1] + cor_landmark106[63]
                   [1] + cor_landmark106[68][1] + cor_landmark106[66][1]) / 4
        w_eye = math.sqrt((x_left - x_right) ** 2 + (y_left - y_right) ** 2)
        crop_w = int(3. * w_eye / 2. + 0.5)
        crop_h = int(3. * crop_w / 2 + 0.5)

        right_start = (np.array([cor_landmark106[75][0] -
                                 (1.62 *
                                  crop_w +
                                  1), cor_landmark106[75][1] -
                                 0.1 *
                                 crop_h]) +
                       0.5).astype(np.int64)
        left_start = (
            np.array(
                [
                    cor_landmark106[76][0],
                    cor_landmark106[76][1] -
                    0.1 *
                    crop_h]) +
            0.5).astype(
                np.int64)

        cor_landmark106_right = cor_landmark106 - right_start
        cor_landmark106_left = cor_landmark106 - left_start

        return [cor_landmark106_right, cor_landmark106_left, int(
            0.5 + crop_w * 1.62 + 1), int(0.5 + crop_h * 1.2)]

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

        M = np.array([M01, M02, M03, M04, M11, M12, M13, M14, M21,
                     M22, M23, M24, M31, M32, M33, M34]).reshape(4, 4)
        x1 = solve(M, np.array([M05 * -1, M15 * -1, M25 * -1, M35 * -1]))
        x = np.array([x1[0], x1[1] * -1, x1[2], x1[1],
                     x1[0], x1[3]]).reshape(2, 3)
        return x

    def make_dir(self, img_path):
        """
        make dir
        :param img_path:
        :return:
        """
        dir_path = img_path.replace(img_path.split('/')[-1], '')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

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
        if self.mat_inter(box1, box2):
            x01, y01, x02, y02 = box1
            x11, y11, x12, y12 = box2

            col = min(x02, x12) - max(x01, x11)
            row = min(y02, y12) - max(y01, y11)
            intersection = col * row
            area1 = (x02 - x01) * (y02 - y01)
            area2 = (x12 - x11) * (y12 - y11)
            coincide = float(intersection) / \
                float((area1 + area2) - intersection)
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

        return suppress

    def cover_face(self, points, faceH, frame, cover=False):
        """
        cover face
        :param points:
        :param faceH:
        :param frame:
        :param cover:
        :return:
        """
        xiaba = points[16]
        zuichun_xiazhong = points[91]

        # print('xiaba', xiaba)
        # print('zuichun_xiazhong', zuichun_xiazhong)
        # cv2.circle(frame_copy, (xiaba[0], xiaba[1]), 1, (255, 255, 0), 2)
        # cv2.circle(frame_copy, (zuichun_xiazhong[0], zuichun_xiazhong[1]), 1, (255, 255, 0), 2)

        if cover == '1_2':
            coverY = int(
                zuichun_xiazhong[1] +
                float(
                    xiaba[1] -
                    zuichun_xiazhong[1]) /
                float(2))  # xiaba 1/2
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

    def predict_images(
            self,
            faceid_model_backbone,
            faceid_model_path,
            onnx_file_faceDetect,
            onnx_facept,
            image_dirs,
            txt_path,
            compare_type,
            log_name,
            cover=False, use_qat=False, use_onnx=False):
        """
        :param net:
        :param path:
        :return:
        """

        for images_path in image_dirs:
            if not os.path.exists(images_path):
                print("{} does not exist".format(images_path))
                exit()

        if not os.path.exists(txt_path):
            os.makedirs(txt_path)

        if compare_type == 0 or compare_type == 1:
            dataset_rgb = get_dataset(image_dirs[0])
            dataset_ir = get_dataset(image_dirs[1])

            # get embeddings
            manager = mp.Manager()
            q_data_used_rgb = manager.Queue()
            q_data_used_ir = manager.Queue()
            q_rgb = manager.Queue()
            q_ir = manager.Queue()

            mp1 = mp.Process(
                target=get_embeddings,
                args=(
                    dataset_rgb,
                    onnx_file_faceDetect,
                    onnx_facept,
                    faceid_model_backbone,
                    faceid_model_path,
                    q_rgb,
                    q_data_used_rgb, use_qat, use_onnx))
            mp2 = mp.Process(
                target=get_embeddings,
                args=(
                    dataset_ir,
                    onnx_file_faceDetect,
                    onnx_facept,
                    faceid_model_backbone,
                    faceid_model_path,
                    q_ir,
                    q_data_used_ir, use_qat, use_onnx))

            mp1.start()
            mp2.start()

            mp1.join()
            mp2.join()

            print_with_time("rgb_embs_got_finished")
            print_with_time("ir_embs_got_finished")

            faceid_feats_rgb = q_rgb.get()
            faceid_feats_ir = q_ir.get()
            data_used_rgb = q_data_used_rgb.get()
            data_used_ir = q_data_used_ir.get()

            print('!!faceid_feats_rgb', len(faceid_feats_rgb))
            print('!!faceid_feats_ir', len(faceid_feats_ir))
            print('!!data_used_rgb', len(data_used_rgb))
            print('!!data_used_ir', len(data_used_ir))

            # compare
            if compare_type == 0:
                faceid_feats_rgbir = copy.deepcopy(faceid_feats_rgb)
                data_used_rgbir = copy.deepcopy(data_used_rgb)
                for cls in faceid_feats_ir.keys():
                    if cls not in faceid_feats_rgbir.keys():
                        faceid_feats_rgbir[cls] = copy.deepcopy(
                            faceid_feats_ir[cls])
                    else:
                        embs_ir = copy.deepcopy(faceid_feats_ir[cls].emb)
                        embs_rgb = copy.deepcopy(faceid_feats_rgb[cls].emb)
                        faceid_feats_rgbir[cls].emb = np.concatenate(
                            (embs_ir, embs_rgb), axis=0)
                    faceid_feats_rgbir[cls].emb_mean = np.mean(
                        faceid_feats_rgbir[cls].emb, axis=0)

                for cls in data_used_ir.keys():
                    if cls not in data_used_rgbir.keys():
                        data_used_rgbir[cls] = copy.deepcopy(data_used_ir[cls])
                    else:
                        image_paths_ir = copy.deepcopy(
                            data_used_ir[cls].image_paths)
                        data_used_rgbir[cls].image_paths += image_paths_ir

                p1 = mp.Process(
                    target=compare_vecs,
                    args=(
                        data_used_ir,
                        faceid_feats_ir,
                        txt_path,
                        'ir',
                        log_name))
                p2 = mp.Process(
                    target=compare_vecs,
                    args=(
                        data_used_rgb,
                        faceid_feats_rgb,
                        txt_path,
                        'rgb',
                        log_name))

                p3 = mp.Process(
                    target=compare_vecs,
                    args=(
                        data_used_rgbir,
                        faceid_feats_rgbir,
                        txt_path,
                        'ir_rgb_merge',
                        log_name))

                p1.start()
                p2.start()
                p3.start()

                p1.join()
                p2.join()
                p3.join()

            elif compare_type == 1:  # two datasets compare
                compare_vecs_twosets(
                    data_used_rgb,
                    faceid_feats_rgb,
                    data_used_ir,
                    faceid_feats_ir,
                    txt_path,
                    'twosets',
                    log_name)

        elif compare_type == 2:  # one dataset compare
            dataset_rgb = get_dataset(image_dirs[0])
            manager = mp.Manager()
            q_data_used_rgb = manager.Queue()
            q_rgb = manager.Queue()
            mp1 = mp.Process(
                target=get_embeddings,
                args=(
                    dataset_rgb,
                    onnx_file_faceDetect,
                    onnx_facept,
                    faceid_model_backbone,
                    faceid_model_path,
                    q_rgb,
                    q_data_used_rgb, use_qat, use_onnx))
            mp1.start()
            mp1.join()
            faceid_feats_rgb = q_rgb.get()
            data_used_rgb = q_data_used_rgb.get()
            print('!!faceid_feats_rgb', len(faceid_feats_rgb))
            print('!!data_used_rgb', len(data_used_rgb))
            compare_vecs(
                data_used_rgb,
                faceid_feats_rgb,
                txt_path,
                'self',
                log_name)

        print_with_time('End!')


FaceID = FaceID()

thresh_range = [0.35, 0.80]
crop_or_evaluate = 1  # 0: crop, 1: evaluate   # 1
use_crop = 1  # 0: use cropped images, 1: crop currently   # 1
batch_size = 1  # onnx: 1

use_onnx = True
use_qat = False
use_prun = False
sen_threshold = 0.5

def main():
    # onnx_file_faceDetect = "./FaceDetect/FaceDetection0816V4Main_noquant.onnx"
    # onnx_facept = "./FaceLandmark/FacialLandmark220816V25Main_noquant.onnx"
    onnx_file_faceDetect = "./assets/FaceDetection20230315V4Main_noquant.onnx"
    onnx_facept = "./assets/FacialLandmark230315V26Main_noquant.onnx"
    sys.path.append('../')

    # faceid_model_backbone = '1012V1_5'  #1012V1  1012V1_5  d5e2a453
    faceid_model_backbone = '1012V1_1024'  # 1012V1_1024    1012V1_512  V7  d5e2a453
    # faceid_model_path_list = ['./model/FaceRecognize220823V15MainNOQAT_sim_skipbn.onnx']
    faceid_model_path_list = ['./assets/FaceRecognize230320V16MainNoQAT_sim_skipbn.onnx']
    # -------------scale------------------
    # faceid_model_path_list = ['./model/hardswish_scale_V15/FaceRecognize220823V15MainNOQAT']

    
    # ---------------run image ori--------------------'
    # [mode, name, dataset1, dataset2], mode:0:standard, 1:two datasets, 2:self
    image_dir_origin_list = []
    root_path = '/media/baidu/a2c56b1b-e2be-44bc-96ba-b4440264cd84/iovfaceid'
#     image_dir_rgb_origin = root_path + '/datasets/hengda_shouji/imgs_big_phone_car/face_feature_phone_rot90_ID'
#     image_dir_ir_origin =  root_path + '/datasets/hengda_shouji/imgs_big_phone_car/face_feature_car'  # diku 9 images
    # image_dir_rgb_origin = root_path + '/datasets/faceIDold/desai_danger_source_resize_clean/rgb'
    # image_dir_ir_origin = root_path + '/datasets/faceIDold/desai_danger_source_resize_clean/ir'
    image_dir_ir_origin =  '/home/frewen/03.ProgramSpace/20.AIStudy/01.WorkSpace/NyxAILearning/AliceInference/baidu_face_detection/python/res/face_feature/ir/'
    image_dir_rgb_origin = '/home/frewen/03.ProgramSpace/20.AIStudy/01.WorkSpace/NyxAILearning/AliceInference/baidu_face_detection/python/res/face_feature/rgb/'
    # image_dir_origin_list.append([0, 'standard', image_dir_rgb_origin, image_dir_ir_origin])
    image_dir_origin_list.append([1, 'two datasets', image_dir_rgb_origin, image_dir_ir_origin])
#     image_dir_origin_list.append([2, 'self', image_dir_rgb_origin])
    
#     # ---------------run image crop--------------------'
#     # [mode, name, dataset1, dataset2], mode:0:standard, 1:two datasets, 2:self
#     image_dir_origin_list = []
#     root_path = '/media/baidu/a2c56b1b-e2be-44bc-96ba-b4440264cd84/iovfaceid'
# #     image_dir_rgb_origin = root_path + '/datasets/weisen_new/desai90_weisen_nobefore_merge'  #95.2649, 0.010921(0.455)
# #     image_dir_ir_origin = root_path + '/datasets/weisen_new/desai90_weisen_nobefore_merge'  # 90.2648, 0.001157(0.395)
# #     image_dir_rgb_origin = root_path + '/datasets/weisen_new/desai90_weisen_nobefore_car3_merge' #95.9550, 0.010647(0.465)
# #     image_dir_ir_origin = root_path + '/datasets/weisen_new/desai90_weisen_nobefore_car3_merge'  #91.1764, 0.001008(0.400) 
#     image_dir_rgb_origin = root_path + '/datasets/weisen_new/desai90_weisen_nobefore_car3_merge_221010QA'#98.3795, 0.011933(0.540) 
#     image_dir_ir_origin = root_path + '/datasets/weisen_new/desai90_weisen_nobefore_car3_merge_221010QA' #96.7521, 0.001226(0.485) 
# #     image_dir_rgb_origin = root_path + '/datasets/weisen_new/desai90_weisen_merge_221010QA'  #  97.6239, 0.011143(0.535)
# #     image_dir_ir_origin = root_path + '/datasets/weisen_new/desai90_weisen_merge_221010QA'   #   94.6416, 0.001187(0.480)
# #     image_dir_rgb_origin = root_path + '/datasets/weisen_new/weisen_video_original_nobadcase' #88.5753, 0.011910(0.425)
# #     image_dir_ir_origin = root_path + '/datasets/weisen_new/weisen_video_original_nobadcase'  #80.0903, 0.001023(0.375)
    
# #     image_dir_rgb_origin = root_path + '/datasets/faceIDold/80ID_series/80IDcrop_norepeat_new/RGB'
# #     image_dir_ir_origin = root_path + '/datasets/faceIDold/80ID_series/80IDcrop_norepeat_new/IR'
# #     image_dir_rgb_origin = root_path + '/datasets/faceIDold/desai_danger_crop_clean_220901/rgb'
# #     image_dir_ir_origin = root_path + '/datasets/faceIDold/desai_danger_crop_clean_220901/ir'
#     image_dir_origin_list.append([0, 'standard', image_dir_rgb_origin, image_dir_ir_origin])


    for faceid_model_path in faceid_model_path_list:
        cover = 'False'  # '1_2'  '1_8' #False
        output_paths = './output_evaluates/'
        log_name = faceid_model_path.split('/')[-1]#.split('.')[0]
        for j in range(len(image_dir_origin_list)):
            compare_type = image_dir_origin_list[j][0]
            TestDay = time.strftime("%m-%d-%Y", time.localtime(time.time()))
            output_path = output_paths + '/' + TestDay + '_ecloud_868_HardSwish_' + \
                faceid_model_backbone + '_' + image_dir_origin_list[j][1] + '/'
            image_dirs = image_dir_origin_list[j][2:]
            FaceID.predict_images(
                faceid_model_backbone,
                faceid_model_path,
                onnx_file_faceDetect,
                onnx_facept,
                image_dirs,
                output_path,
                compare_type,
                log_name,
                cover, use_qat, use_onnx)

    # ---------------run camera--------------------'
    # camera
    #FaceID.predict_camera(net, ort_sess_faceDetect, ort_sess_faceLandmark, use_onnx_model)

    print('faceid_model_path', faceid_model_path)


if __name__ == '__main__':
    dist.spawn(main)   # run: Train.sh
