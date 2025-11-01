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
    allFilesPath1 = getAllFiles(images_path)
    allFilesPath = []
    # for item in allFilesPath1:
    #     if "/drink/" in item or "/normal/" in item or "/silence/" in item or "/smoke/" in item or "/yawn/" in item:
    #         allFilesPath.append(item)
    allFilesPath = allFilesPath1
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
        image_origin = cv2.resize(image, (288, 160))
        image_origin_rgb = cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB)
        image_pre = (image_origin_rgb.astype(np.float32) / 255. - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        # print('resized image shape:', image_pre.shape)
        image = np.expand_dims(image_pre, axis=0)

        image = (image.astype(np.float32))
        image = np.transpose(image, [0, 3, 1, 2])

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
                # if pred[0] == 0:
                #     cv2.rectangle(image_disp, (y1, x1), (y2, x2), color=(255, 0, 0), thickness=3)
                # elif pred[0] == 1:
                #     cv2.rectangle(image_disp, (y1, x1), (y2, x2), color=(0, 255, 0), thickness=3)
                # else:
                #     cv2.rectangle(image_disp, (y1, x1), (y2, x2), color=(0, 0, 255), thickness=3)
                f_preScore = pred[1].numpy()[0]
                f_preScore = round(f_preScore, 5)
                cv2.putText(image_disp, str(round(f_preScore,3)), (y1, x1),1, 1, (0,0,255), 1, 8, 0)
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
        cap.set(3, 1280)
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
                        if pred[1] >= 0.2:
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
                            fscore = round(fscore[0], 3)
                            cv2.putText(image_disp, str(fscore)+"|"+str(y2-y1)+"|"+str(x2-x1), (y1,x1), 1, 3, (0,0,255), 3, 8, 0)
                    if write_video:
                        out.write(image_disp)
                    cv2.imshow('preds', image_disp)
                    cv2.waitKey(1)

from reader import TrainReader, EvalReader
from vehicle_pdc_hand import HandDataSet
from paddle.static import InputSpec

if __name__ == '__main__':
    net = PPYoloTiny(model="2d96")
    # net = PPYoloTiny(model="mbv3")

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
        pruner.sensitive(
            sen_file="/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/剪枝文件/L1Norm_sen_yb_sense60.pickle")
            # sen_file="/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/芯算一体/model50M224/剪枝文件/L1Norm_sen_yb_sense2.pickle")
            # sen_file = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/芯算一体/c7230302/剪枝文件/L1Norm_sen_yb_sense2.pickle")
        paddle.flops(net, (1, 3, 160, 288))
        print("+1" * 100)
        # plan = pruner.sensitive_prune(0.7) #c7230302
        plan = pruner.sensitive_prune(0.725) #ppyolotiny mbv3
        paddle.flops(net, (1, 3, 160, 288))
        print("+2" * 100)

    # pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/剪枝模型/v2/20211115174924/20211116192715_epoch17_v2.pdparams"
    # pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/剪枝模型/v3/epoch19_v3.pdparams" #ppyolotiny剪枝模型
    # #原始ppyolotiny未剪枝模型,使用此模型时需修改为原来的hardsigmoid  hardswish
    # pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/原始模型/20211108_tmpV2/20211107123752_epoch24_1000_v2.pdparams"
    # pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/原始模型/原始模型v3_未加载预训练backbone/epoch19.pdparams"
    # pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/芯算一体/c7230302/剪枝模型70/epoch19.pdparams"
    # pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/芯算一体/c7230302/剪枝模型85/epoch19.pdparams"
    pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/芯算一体/c7230302/epoch19.pdparams"
    pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/芯算一体/2d4f5904/epoch19.pdparams"
    pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/芯算一体/b0b5e552/epoch19.pdparams"
    pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/芯算一体/2d96_ori/epoch19.pdparams"
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
        ignore_weights = set()

        for name, weight in param_state_dict.items():
            # if name in model_dict.keys():

                # if ".bn" in name:
            print(name, param_state_dict[name].shape, type(param_state_dict[name]))
            print (param_state_dict[name])
                # if list(weight.shape) != list(model_dict[name].shape):
                #     ignore_weights.add(name)
            # else:
            #     ignore_weights.add(name)

        for weight in ignore_weights:
            param_state_dict.pop(weight, None)

        model.set_dict(param_state_dict)
    load_pretrain_weight(net, pathModel)

    # pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/芯算一体/model50M224/prune0.8/epoch1.pdparams"
    layer_state_dict = paddle.load(pathModel)
    net.set_state_dict(layer_state_dict)

    net.eval()

    for name, weight in layer_state_dict.items():
        # if ".bn" in name:
        print(name, layer_state_dict[name].shape, type(layer_state_dict[name]))
        # print(layer_state_dict[name])



    # paddle.jit.save(net, "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/芯算一体/c7230302/剪枝模型55/c7230302_prune55_static/facebox_c7230302_prune0.55", input_spec=[InputSpec(shape=[1,3,160,288], dtype=paddle.float32)] )
    # paddle.jit.save(net, "/media/baidu/UPAN/faceONNX/v2/facebox_c7230302_prune70", input_spec=[InputSpec(shape=[1,3,160,288], dtype=paddle.float32)] )


    # images_path = "/media/baidu/ssd2/ppyolo/6w_data/paddle_car/paddle_car/gesture_demo"
    # images_path = "/media/baidu/3.6TB_SSD/facedetect/验证集数据/val_resize"
    # folder_name1 = "image"
    # folder_name2 = folder_name1 + "_c7230302Prune70_BoxRslt"


    images_path = "/media/baidu/ssd1/dms_data/test_data"
    folder_name1 = "INCAR-IMG"
    folder_name2 = folder_name1 + "_c7230302Prune70_BoxRslt"
    folder_name2 = folder_name1 + "_ppyolotinyPrune_BoxRslt"

    images_path = "/media/baidu/3.6TB_SSD/JiDu_valdata/dmsData_daYaw"
    folder_name1 = "pose_daYaw"
    folder_name2 = folder_name1 + "_ppyolotinyPrune_BoxRslt"

    images_path = "/media/baidu/ssd1/表情"
    folder_name1 = "emotion_eval"
    folder_name2 = folder_name1 + "_paddle_facebox"

    # predict_images(net, os.path.join(images_path, folder_name1))
    # paddle.flops(net, [1,3,160,288])
    predict_video(net, './video', write_video=False)

    # for name in net.state_dict().keys():
    #     print (name, net.state_dict()[name].shape)
