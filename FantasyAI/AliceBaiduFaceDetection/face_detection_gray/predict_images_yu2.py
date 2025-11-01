"""
predict images
"""
import copy

import paddle
import os
from PPYoloMobileNetV3 import PPYoloMobileNetV3, PPYoloTiny
import cv2
import numpy as np
import random


def predict_images_maxmin(net):
    """
    predict images
    :param net: net
    :param images_path:images
    :return: result
    """
    images_path = "/media/baidu/ssd2/ppyolo/data_mp4/facedata/images"
    pathSave_maxmin = "/media/baidu/ssd2/ppyolo/data_mp4/facedata/images_save"
    if not os.path.isdir(pathSave_maxmin):
        os.makedirs(pathSave_maxmin)

    def getallfiles(path):
        allFiles = []
        for root, folder, files in os.walk(path):
            for filet in files:
                if filet.split(".")[-1] in ['jpg', 'jpeg', "JPG", "JPEG", "png", "PNG", "bmp", "BMP"]:
                    allFiles.append(os.path.join(root, filet))
        return allFiles

    files = os.listdir(images_path)
    # files = getallfiles(images_path)
    # random.shuffle(files)
    print('files num:', len(files))
    for file in files:
        # img_file = '/Users/wangyawei05/Documents/wyw/paddle/data/obstacle/image_samples100/BeiSanHuanHengXiangKouDong_532_1563145849000_1563188238000_V07254101_162.jpg'
        # img_full = '/Users/wangyawei05/Documents/wyw/laneline/data/hwp/images_sample/arrow-on-the-ground-jianghuai-normal-02-normal_36-8541-8615_8586.jpg'
        img_full = os.path.join(images_path, file)
        # img_full = file
        print(img_full)
        image = cv2.imread(img_full)
        if image is None:
            continue
        h1,w1,_ = image.shape
        print('original image shape:', image.shape)
        image_disp = image.copy()
        norm_w = 1280
        norm_h = 720
        image_disp = cv2.resize(image_disp, (norm_w, norm_h))
        fratio_x = norm_w * 1.0 / w1
        fratio_y = norm_h * 1.0 / h1
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_origin = cv2.resize(image, (288, 160))
        image_origin_rgb = cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB)
        image_pre = (image_origin_rgb.astype(np.float32) / 255. - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        print('resized image shape:', image_pre.shape)
        image = np.expand_dims(image_pre, axis=0)

        image = (image.astype(np.float32))
        image = np.transpose(image, [0, 3, 1, 2])

        net.eval()
        boxes, num_boxes = net(paddle.to_tensor(image))

        print('preds:', num_boxes)

        for i, pred in enumerate(boxes):
            if i == 0:
                file_rslt = open(os.path.join(pathSave_maxmin, file+".txt"), "w")
            if pred[1] > -1000.1: #0.75:
                y1_float = pred[2] * image_disp.shape[1] / image_origin.shape[1]
                x1_float = pred[3] * image_disp.shape[0] / image_origin.shape[0]
                y2_float = pred[4] * image_disp.shape[1] / image_origin.shape[1]
                x2_float = pred[5] * image_disp.shape[0] / image_origin.shape[0]
                y1 = int(y1_float)
                x1 = int(x1_float)
                y2 = int(y2_float)
                x2 = int(x2_float)
                if pred[0] == 0:
                    cv2.rectangle(image_disp, (y1, x1), (y2, x2), color=(255, 0, 0), thickness=3)
                elif pred[0] == 1:
                    cv2.rectangle(image_disp, (y1, x1), (y2, x2), color=(0, 255, 0), thickness=3)
                else:
                    cv2.rectangle(image_disp, (y1, x1), (y2, x2), color=(0, 0, 255), thickness=3)
                f_preScore = pred[1].numpy()[0]
                f_preScore = round(f_preScore, 3)
                cv2.putText(image_disp, str(round(f_preScore,3)), (y1, x1),1, 1, (0,0,255), 1, 8, 0)
                box_w = (pred[4] - pred[2]) * norm_w / image_origin.shape[1]
                box_h = (pred[5] - pred[3]) * norm_h / image_origin.shape[0]
                box_w = box_w.numpy()
                box_h = box_h.numpy()
                box_w = str(round(box_w[0], 2))
                box_h = str(round(box_h[0], 2))
                cv2.putText(image_disp, "w=" + str(box_w) + ",h=" + str(box_h), (y1 + 50, x1), 1, 1, (0, 0, 255), 1, 8, 0)
                file_rslt.write(str(f_preScore)+"\t" + str(y1) + "\t" + str(x1) + "\t" + str(y2) + "\t" + str(x2) + "\n")

        cv2.imwrite(os.path.join(pathSave_maxmin, file), image_disp)
        # cv2.imshow('preds', image_disp)
        # cv2.waitKey(1)

def predict_images(net):
    """
    predict images
    :param net: net
    :param images_path:images
    :return: result
    """
    images_path = "/media/baidu/3.6TB_SSD/facedetect/验证集数据/val_原始数据/image"
    pathSave = "/media/baidu/3.6TB_SSD/facedetect/验证集数据/val_原始数据/rslt/rslt_prune"

    images_path = "/media/baidu/ssd1/标注数据/second/norm_img/valdata"
    pathSave = "/media/baidu/ssd1/标注数据/second/norm_img/valdata_drawrslt"

    if not os.path.isdir(pathSave):
        os.makedirs(pathSave)

    def getallfiles(path):
        allFiles = []
        for root, folder, files in os.walk(path):
            for filet in files:
                if filet.split(".")[-1] in ['jpg', 'jpeg', "JPG", "JPEG", "png", "PNG", "bmp", "BMP"]:
                    allFiles.append(os.path.join(root, filet))
        return allFiles

    files = os.listdir(images_path)
    # files = getallfiles(images_path)
    # random.shuffle(files)
    print('files num:', len(files))
    for file in files:
        # img_file = '/Users/wangyawei05/Documents/wyw/paddle/data/obstacle/image_samples100/BeiSanHuanHengXiangKouDong_532_1563145849000_1563188238000_V07254101_162.jpg'
        # img_full = '/Users/wangyawei05/Documents/wyw/laneline/data/hwp/images_sample/arrow-on-the-ground-jianghuai-normal-02-normal_36-8541-8615_8586.jpg'
        img_full = os.path.join(images_path, file)
        # img_full = file
        print(img_full)
        image = cv2.imread(img_full)
        if image is None:
            continue
        print('original image shape:', image.shape)
        image_disp = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_origin = cv2.resize(image, (288, 160))
        image_origin_rgb = cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB)
        image_pre = (image_origin_rgb.astype(np.float32) / 255. - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        print('resized image shape:', image_pre.shape)
        image = np.expand_dims(image_pre, axis=0)

        image = (image.astype(np.float32))
        image = np.transpose(image, [0, 3, 1, 2])

        net.eval()
        boxes, num_boxes = net(paddle.to_tensor(image))

        print('preds:', num_boxes)

        for i, pred in enumerate(boxes):
            if i == 0:
                file_rslt = open(os.path.join(pathSave, file+".txt"), "w")
            if pred[1] > 0.2: #0.75:
                y1 = int(pred[2] * image_disp.shape[1] / image_origin.shape[1])
                x1 = int(pred[3] * image_disp.shape[0] / image_origin.shape[0])
                y2 = int(pred[4] * image_disp.shape[1] / image_origin.shape[1])
                x2 = int(pred[5] * image_disp.shape[0] / image_origin.shape[0])
                if pred[0] == 0:
                    cv2.rectangle(image_disp, (y1, x1), (y2, x2), color=(255, 0, 0), thickness=3)
                elif pred[0] == 1:
                    cv2.rectangle(image_disp, (y1, x1), (y2, x2), color=(0, 255, 0), thickness=3)
                else:
                    cv2.rectangle(image_disp, (y1, x1), (y2, x2), color=(0, 0, 255), thickness=3)
                f_preScore = pred[1].numpy()[0]
                f_preScore = round(f_preScore, 3)
                cv2.putText(image_disp, str(round(f_preScore,3)), (y1, x1),1, 1, (0,0,255), 1, 8, 0)
                cv2.putText(image_disp, "w=" + str(y2 - y1) + ",h=" + str(x2 - x1), (y1 + 45, x1), 1, 1, (0, 0, 255), 1, 8, 0)
                file_rslt.write(str(f_preScore)+"\t" + str(y1) + "\t" + str(x1) + "\t" + str(y2) + "\t" + str(x2) + "\n")

        cv2.imwrite(os.path.join(pathSave, file), image_disp)
        # cv2.imshow('preds', image_disp)
        # cv2.waitKey(1)

def predict_video(net, path, write_video=False):
    """

    :param net:
    :param path:
    :return:
    """
    # files = os.listdir(path)
    # random.shuffle(files)
    # print('files num:', len(files))
    # for file in files:

    # Video_full_path = os.path.join(path, file)
    # cap = cv2.VideoCapture(Video_full_path)
    file = None
    # cap = cv2.VideoCapture("/home/baidu/Desktop/video2/Video_2021-10-27_13-31-17.mp4")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, image = cap.read()
        if write_video:
            Video_full_out_path = os.path.join(path, 'out_' + file)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(Video_full_out_path, fourcc, 30.0, (image.shape[1], image.shape[0]), True)

        nframe_id = 0
        while (ret):
            ret, image = cap.read()
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if ret:
                if image is None:
                    continue
                # img2 = np.zeros(shape=(720,1280,3), dtype=np.uint8)
                # img2[:480, :640, :] = copy.deepcopy(image)
                # image = copy.deepcopy(img2)

                # print('original image shape:', image.shape)
                image_disp = image.copy()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_origin = cv2.resize(image, (288, 160))
                image_pre = (image_origin.astype(np.float32) / 255. - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                # print('resized image shape:', image_pre.shape)
                image = np.expand_dims(image_pre, axis=0)

                image = (image.astype(np.float32))
                image = np.transpose(image, [0, 3, 1, 2])

                net.eval()
                boxes, num_boxes = net(paddle.to_tensor(image))

                # print('preds:', num_boxes)
                for i, pred in enumerate(boxes):
                    if pred[1] >= 0.3:
                        y1 = int(pred[2] * image_disp.shape[1] / image_origin.shape[1])
                        x1 = int(pred[3] * image_disp.shape[0] / image_origin.shape[0])
                        y2 = int(pred[4] * image_disp.shape[1] / image_origin.shape[1])
                        x2 = int(pred[5] * image_disp.shape[0] / image_origin.shape[0])
                        if pred[0] == 0:
                            cv2.rectangle(image_disp, (y1, x1), (y2, x2), color=(255, 0, 0), thickness=3)
                        elif pred[0] == 1:
                            cv2.rectangle(image_disp, (y1, x1), (y2, x2), color=(0, 255, 0), thickness=3)
                        else:
                            cv2.rectangle(image_disp, (y1, x1), (y2, x2), color=(0, 0, 255), thickness=3)
                        fscore = pred[1].numpy()[0]
                        print ("w=",y2-y1, "h=",x2-x1)
                        cv2.putText(image_disp, str(round(fscore,2)), (y1,x1), 1, 1, (0,0,255), 1, 8, 0)
                        cv2.putText(image_disp, "w="+str(y2-y1)+",h="+str(x2-x1), (y1+30, x1), 1, 2, (0,0,255), 2, 8, 0)
                        nframe_id += 1
                        cv2.imwrite("/home/baidu/Desktop/tmpyu/face/" + str(nframe_id).zfill(6) + ".jpg", image_disp)
                if write_video:
                    out.write(image_disp)
                cv2.imshow('preds', image_disp)
                cv2.waitKey(1)

from reader import TrainReader, EvalReader
from vehicle_pdc_hand import HandDataSet

if __name__ == '__main__':
    # net = PPYoloTiny(model="2d96")
    # net = PPYoloTiny(model="2d4f5904")
    # net = PPYoloTiny(model="792bcf52")
    # net = PPYoloTiny(model="b0b5e552")
    # net = PPYoloTiny(model="model50M224")
    net = PPYoloTiny(model="c7230302")

    # for name in net.state_dict().keys():
    #     print (name, net.state_dict()[name].shape)
    # paddle.summary(net, (1,3,160,320))
    # print ("1"*200)
    # paddle.flops(net, (1,3,160,320))
    usePruner = 0
    if usePruner:
        try:
            from paddleslim.dygraph import L1NormFilterPruner
        except:
            os.system('pip3 install paddleslim')
            from paddleslim.dygraph import L1NormFilterPruner
        #         from paddleslim.dygraph import L1NormFilterPruner
        pruner = L1NormFilterPruner(net, [1, 3, 160, 288])
        # sen = pruner.sensitive(eval_func=eval_fn, sen_file=model_save_path + "L1Norm_sen_yb_sense2.pickle", skip_vars=['conv2d_80.w_0','conv2d_80.w_0','conv2d_80.w_0'])
        # pruner.sensitive( sen_file="/media/baidu/ssd2/ppyolo/6w_data/paddle_car/paddle_car/model_hand/20210926191010L1Norm_sen_yb_sense3.pickle")
        pruner.sensitive(sen_file="/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/剪枝文件/L1Norm_sen_yb_sense60.pickle")
        paddle.flops(net, (1, 3, 160, 288))
        print("+1" * 100)
        plan = pruner.sensitive_prune(0.725)
        paddle.flops(net, (1, 3, 160, 288))
        print("+2" * 100)

    #2d96
    pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/芯算一体/2d96_ori/epoch19.pdparams"
    #2d4f5904
    pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/芯算一体/2d4f5904/epoch19.pdparams"
    #792bcf52
    pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/芯算一体/792bcf52/epoch19.pdparams"
    #b0b5e552
    pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/芯算一体/b0b5e552/epoch19.pdparams"
    #model50M224
    pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/芯算一体/model50M224/epoch19.pdparams"
    #c7230302
    pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/芯算一体/c7230302/epoch19.pdparams"

    layer_state_dict = paddle.load(pathModel)
    net.set_state_dict(layer_state_dict)

    paddle.flops(net, (1, 3, 160, 288))
    # images_path = '/Users/leisheng526/Development/baidu/iov-bj/base-lane-lines-finding/hwp/' \
    #               'images/sand-storm-jianghuai-normal-03-curve_30-1487-1583'



    # predict_images_maxmin(net)
    #

    # predict_images(net)
    #
    predict_video(net, './video', write_video=False)

    # for name in net.state_dict().keys():
    #     print (name, net.state_dict()[name].shape)