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
    allFilesPath = getAllFiles(images_path)
    # random.shuffle(files)
    print('files num:', len(allFilesPath))
    nid = 0
    for filepath in allFilesPath:
        if "/face_detection_" in filepath:
            continue

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
        h1,w1,_ = image.shape
        w1_new = w1
        if h1>w1:
            w1_new = int(h1*1.7777)
        image_norm2 = np.zeros(shape=[h1,w1_new, 3], dtype=np.uint8)
        image_norm2[:h1, :w1, :] = image
        image = image_norm2

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
                if pred[0] == 0:
                    cv2.rectangle(image_disp, (y1, x1), (y2, x2), color=(255, 0, 0), thickness=3)
                elif pred[0] == 1:
                    cv2.rectangle(image_disp, (y1, x1), (y2, x2), color=(0, 255, 0), thickness=3)
                else:
                    cv2.rectangle(image_disp, (y1, x1), (y2, x2), color=(0, 0, 255), thickness=3)
                f_preScore = pred[1].numpy()[0]
                f_preScore = round(f_preScore, 3)
                cv2.putText(image_disp, str(round(f_preScore,3)), (y1, x1),1, 1, (0,0,255), 1, 8, 0)
                if x1 > w1 or x2 > w1 or y1 > h1 or y2 > h1:
                    continue
                file_rslt.write(str(f_preScore)+"\t" + str(y1) + "\t" + str(x1) + "\t" + str(y2) + "\t" + str(x2) + "\n")

        cv2.imwrite(pathSave, image_disp)
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
        cap = cv2.VideoCapture(2)
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
    # net = PPYoloTiny("mbv3")
    net = PPYoloTiny("2d4f5904")
    paddle.flops(net, (1, 3, 160, 288))
    print("+1" * 100)

    usePruner = 1
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
        # pruner.sensitive(sen_file="/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/剪枝文件/L1Norm_sen_yb_sense60.pickle")

        #pd222  mbv3剪枝文件
        # pruner.sensitive(sen_file="/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/pd222/剪枝文件/L1Norm_sen_yb_pd222_v2.pickle")
        # pd222  2d4f5904剪枝文件
        pruner.sensitive(sen_file="/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/pd222/2d4f5904/剪枝量化/剪枝文件/xinsuanyiti_2d4f5904_prune.pickle")

        paddle.flops(net, (1, 3, 160, 288))
        # print("+1" * 100)
        plan = pruner.sensitive_prune(0.6)
        paddle.flops(net, (1, 3, 160, 288))
        print("+2" * 100)

    bool_quant = 1
    if bool_quant == 1:
        quant_config = {
            "weight_preprocess_type":None,
            "activation_preprocess_type":None,
            "weight_quantize_type":"abs_max",
            "activation_quantize_type":"moving_average_abs_max",
            "weight_bits":8,
            "activation_bits":8,
            "dtype":"int8",
            "window_size":10000,
            "moving_rate":0.9,
            "quantizable_layer_type":['Conv2D', 'Linear'],
        }
        import paddleslim
        quanter = paddleslim.QAT(config=quant_config)
        quanter.quantize(net)
        paddle.flops(net, (1, 3, 160, 288))
        print("+3" * 100)
    pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/剪枝模型/v2/20211115174924/20211116192715_epoch17_v2.pdparams"
    pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/剪枝模型/v3/epoch19.pdparams" #ppyolotiny剪枝模型
    # #原始ppyolotiny未剪枝模型,使用此模型时需修改为原来的hardsigmoid  hardswish
    # pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/原始模型/20211108_tmpV2/20211107123752_epoch24_1000_v2.pdparams"
    # pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/原始模型/原始模型v3_未加载预训练backbone/epoch19.pdparams"
    #
    # #原始ppyolotiny V1 未添加小人脸， 20211102
    # pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/原始模型/20211102151155_tmpV1/20211102214616_epoch4_1000.pdparams"
    # # 原始ppyolotiny V2 未添加小人脸， 20211108
    # pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/原始模型/20211108_tmpV2/20211107123752_epoch24_1000_v2.pdparams"
    #原始ppyolotiny 剪枝模型 20211111
    # pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/剪枝模型/v1/20211111103031_epoch24_2000.pdparams"

    # pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/pd222/5万数据/原始模型/epoch4.pdparams"
    # pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/pd222/5万数据/剪枝模型/epoch5.pdparams"
    # pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/pd222/5万数据/量化模型/epoch19.pdparams"
    # pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/pd222/全量数据/原始模型/epoch9.pdparams"
    # pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/pd222/全量数据/量化模型/epoch19.pdparams"
    # pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/pd222/全量数据/剪枝模型/epoch4.pdparams"

    pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/pd222/全量数据/量化模型/v3_a100/epoch18_release.pdparams"
    pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/pd222/2d4f5904/剪枝量化/剪枝90量化/训练失败/epoch8.pdparams"
    pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/pd222/2d4f5904/剪枝量化/剪枝70量化/epoch24.pdparams"
    pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/pd222/2d4f5904/剪枝量化/剪枝60量化/epoch24.pdparams"
    # pathModel = "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/pd222/2d4f5904/剪枝量化/剪枝80量化/epoch24.pdparams"

    layer_state_dict = paddle.load(pathModel)
    net.set_state_dict(layer_state_dict)

    net.eval()
    # paddle.jit.to_static(net,  input_spec=[InputSpec(shape=[1,3,160,288], dtype=paddle.float32, name='inputs')])
    # paddle.jit.save(net,"/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/pd222/2d4f5904/剪枝量化/剪枝80量化/static_dim1/model")
    #下面方法已不可行
    # paddle.jit.save(net, "/media/baidu/3.6TB_SSD/facedetect/缩小尺寸/models_save/pd222/5万数据/量化模型/static/ppyolotiny_prune0.725", input_spec=[InputSpec(shape=[1,3,160,288], dtype=paddle.float32)] )


    # images_path = "/media/baidu/ssd2/ppyolo/6w_data/paddle_car/paddle_car/gesture_demo"
    images_path = "/media/baidu/3.6TB_SSD/valdata_toQA/facebox/ori"
    folder_name1 = "106测试集"
    folder_name2 = folder_name1 + "_QA_norm"

    images_path = "/media/baidu/3.6TB_SSD/valdata_toQA/facebox/fromQA"
    folder_name1 = "face_detection"
    folder_name2 = folder_name1 + "_QA_noFace"

    # images_path = "/media/baidu/ssd2/本田23M/examples"
    # folder_name1 = "data_mqf"
    # folder_name2 = folder_name1 + "_test量化_20220217"

    # predict_images(net, os.path.join(images_path, folder_name1))
    # paddle.flops(net, [1,3,160,288])
    # predict_video(net, './video', write_video=False)

    # for name in net.state_dict().keys():
    #     print (name, net.state_dict()[name].shape)
