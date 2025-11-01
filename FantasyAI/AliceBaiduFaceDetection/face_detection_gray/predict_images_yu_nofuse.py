"""
predict images
"""
import time
import datetime
import paddle
import os
from PPYoloMobileNetV3 import PPYoloMobileNetV3, PPYoloTiny
import cv2
import numpy as np
import random

def load_pretrain_weight(model, weight):
    """
    加载预训练模型
    :param model:
    :param weight:
    :return:
    """
    weights_path = weight

    if not (os.path.exists(weights_path)):
        raise ValueError("Model pretrain path `{}` does not exists. "
                         "If you don't want to load pretrain model, "
                         "please delete `pretrain_weights` field in "
                         "config file.".format(weights_path))

    model_dict = model.state_dict()
    param_state_dict = paddle.load(weights_path)
    param_state_dict_fuse = {}
    ignore_weights = set()

    for name, weight in param_state_dict.items():
        if name in model_dict.keys():
            if list(weight.shape) != list(model_dict[name].shape):
                ignore_weights.add(name)
        else:
            ignore_weights.add(name)
        if "backbone." + name in model_dict:
            param_state_dict_fuse["backbone." + name] = weight

    for weight in ignore_weights:
        param_state_dict.pop(weight, None)
    model.set_dict(param_state_dict_fuse)

def getAllFiles(path):
    allPathImgs = []
    for root, folder, files in os.walk(path):
        for filet in files:
            if filet.split(".")[-1] in ["jpg", "jpeg", "png", "bmp", "JPG"]:
                allPathImgs.append(os.path.join(root, filet))
    return allPathImgs

def predict_images(net, images_path):
    """
    predict images
    :param net: net
    :param images_path:images
    :return: result
    """
    # files = os.listdir(images_path)
    allFilesPath = getAllFiles(images_path)
    # random.shuffle(files)
    print('files num:', len(allFilesPath))
    nid = 0
    for filepath in allFilesPath:
        nid += 1
        if nid % 1000 == 0:
            print (datetime.datetime.now(), str(nid) + "/" + str(len(allFilesPath)))
            print ("*"*100)
        # img_file = '/Users/wangyawei05/Documents/wyw/paddle/data/obstacle/image_samples100/BeiSanHuanHengXiangKouDong_532_1563145849000_1563188238000_V07254101_162.jpg'
        # img_full = '/Users/wangyawei05/Documents/wyw/laneline/data/hwp/images_sample/arrow-on-the-ground-jianghuai-normal-02-normal_36-8541-8615_8586.jpg'

        print(filepath)
        image = cv2.imread(filepath)
        if image is None:
            continue
        # print('original image shape:', image.shape)
        image_disp = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_origin = cv2.resize(image, (320, 224))
        image_origin_rgb = cv2.cvtColor(image_origin, cv2.COLOR_BGR2GRAY)
        image_pre = image_origin_rgb.astype(np.float32)# / 255. - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        # print('resized image shape:', image_pre.shape)
        image = np.expand_dims(image_pre, axis=0)
        image = np.expand_dims(image, axis=0)

        image = (image.astype(np.float32))
        # image = np.transpose(image, [0, 3, 1, 2])

        net.eval()
        boxes, num_boxes = net(paddle.to_tensor(image))

        # print('preds:', num_boxes)

        for i, pred in enumerate(boxes):
            if i == 0:
                pathSave = filepath.replace(folder_name1, folder_name2)
                pathInfos = pathSave.split("/")
                pathPrefix = "/".join(pathInfos[:-1])
                if not os.path.isdir(pathPrefix):
                    os.makedirs(pathPrefix)
                file_rslt = open(pathSave +".txt", "w")
            if pred[1] > -1000.1: #0.75:
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
                cv2.imwrite(pathSave, image_disp)
                file_rslt.write(str(f_preScore)+"\t" + str(y1) + "\t" + str(x1) + "\t" + str(y2) + "\t" + str(x2) + "\n")

        # cv2.imwrite(os.path.join(pathSave, file), image_disp)
        # cv2.imshow('preds', image_disp)
        # cv2.waitKey(0)

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
    for i in range(10):
        # Video_full_path = os.path.join(path, file)
        # cap = cv2.VideoCapture(Video_full_path)
        file = None
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)  # 设置分辨率
        cap.set(4, 720)
        if cap.isOpened():
            ret, image = cap.read()
            if write_video:
                Video_full_out_path = os.path.join(path, 'out_' + file)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(Video_full_out_path, fourcc, 30.0, (image.shape[1], image.shape[0]), True)

            while (ret):
                ret, image = cap.read()
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if ret:
                    if image is None:
                        continue
                    # print('original image shape:', image.shape)
                    image_disp = image.copy()
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image_origin = cv2.resize(image, (320, 224))
                    image_pre = image_origin.astype(np.float32)  / 255. #- [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                    # print('resized image shape:', image_pre.shape)
                    image = np.expand_dims(image_pre, axis=0)
                    image = np.expand_dims(image, axis=0)

                    image = (image.astype(np.float32))
                    # image = np.transpose(image, [0, 3, 1, 2])

                    net.eval()
                    boxes, num_boxes = net(paddle.to_tensor(image))

                    # print('preds:', num_boxes)
                    for i, pred in enumerate(boxes):
                        if pred[1] > 0.3:
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
                            fscore = pred[1].numpy()
                            facew = y2-y1
                            faceh = x2-x1
                            fscore = round(fscore[0], 3)
                            cv2.putText(image_disp, str(fscore)+"|"+str(facew) + "|" + str(faceh), (y1,x1), 1, 3, (0,0,255), 3, 8, 0)
                    if write_video:
                        out.write(image_disp)
                    cv2.imshow('preds', image_disp)
                    cv2.waitKey(1)

from reader import TrainReader, EvalReader
from vehicle_pdc_hand import HandDataSet
from paddle.static import InputSpec

if __name__ == '__main__':
    net = PPYoloTiny("2d4f5904")
    usePruner = 1
    if usePruner:
        try:
            from paddleslim.dygraph import L1NormFilterPruner
        except:
            os.system('pip3 install paddleslim')
            from paddleslim.dygraph import L1NormFilterPruner
        #         from paddleslim.dygraph import L1NormFilterPruner
        pruner = L1NormFilterPruner(net, [1, 1, 160, 288])
        # sen = pruner.sensitive(eval_func=eval_fn, sen_file=model_save_path + "L1Norm_sen_yb_sense2.pickle", skip_vars=['conv2d_80.w_0','conv2d_80.w_0','conv2d_80.w_0'])
        # pruner.sensitive( sen_file="/media/baidu/ssd2/ppyolo/6w_data/paddle_car/paddle_car/model_hand/20210926191010L1Norm_sen_yb_sense3.pickle")
        # pruner.sensitive(sen_file="/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/剪枝文件/L1Norm_sen_yb_sense60.pickle")
        #pd222
        pruner.sensitive(sen_file="/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/pd222/2d4f5904/1107V4/nofuse/dan_2d4f_prune.pickle")

        paddle.flops(net, (1, 1, 160, 288))
        print("+1" * 100)
        plan = pruner.sensitive_prune(0.5)
        paddle.flops(net, (1, 1, 160, 288))
        print("+2" * 100)
    #pd222  v2
    # pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/pd222/全量数据/剪枝模型/v2/epoch19.pdparams"
    pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/pd222/2d4f5904/1107V4/nofuse/epoch12_using.pdparams"
    layer_state_dict = paddle.load(pathModel)
    net.set_state_dict(layer_state_dict)
    # load_pretrain_weight(net, pathModel)
    net.eval()
    # paddle.jit.save(net, "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/剪枝模型/v3/ppyolotiny_prune0.725_static/ppyolotiny_prune0.725", input_spec=[InputSpec(shape=[1,3,160,288], dtype=paddle.float32)] )


    # images_path = "/media/baidu/ssd2/ppyolo/6w_data/paddle_car/paddle_car/gesture_demo"
    images_path = "/media/baidu/3.6TB_SSD/facedetect/验证集数据/val_resize"
    folder_name1 = "image"
    folder_name2 = folder_name1 + "_原始数据"

    # predict_images(net, os.path.join(images_path, folder_name1))
    # paddle.flops(net, [1,3,160,288])
    predict_video(net, './video', write_video=False)

    # for name in net.state_dict().keys():
    #     print (name, net.state_dict()[name].shape)
