"""
predict images
"""
import paddle
import os
from PPYoloMobileNetV3 import PPYoloMobileNetV3, PPYoloTiny
import cv2
import numpy as np
import random


def predict_images(net, images_path):
    """
    predict images
    :param net: net
    :param images_path:images
    :return: result
    """
    files = os.listdir(images_path)
    random.shuffle(files)
    print('files num:', len(files))
    for file in files:
        # img_file = '/Users/wangyawei05/Documents/wyw/paddle/data/obstacle/image_samples100/BeiSanHuanHengXiangKouDong_532_1563145849000_1563188238000_V07254101_162.jpg'
        # img_full = '/Users/wangyawei05/Documents/wyw/laneline/data/hwp/images_sample/arrow-on-the-ground-jianghuai-normal-02-normal_36-8541-8615_8586.jpg'
        img_full = os.path.join(images_path, file)
        print(img_full)
        image = cv2.imread(img_full)
        if image is None:
            continue
        print('original image shape:', image.shape)
        image_disp = image.copy()
        image_origin = cv2.resize(image, (320, 160))
        image_pre = (image_origin.astype(np.float32) / 255. - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        print('resized image shape:', image_pre.shape)
        image = np.expand_dims(image_pre, axis=0)

        image = (image.astype(np.float32))
        image = np.transpose(image, [0, 3, 1, 2])

        net.eval()
        boxes, num_boxes = net(paddle.to_tensor(image))

        print('preds:', num_boxes)
        for i, pred in enumerate(boxes):
            if pred[1] > 0.75:
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

        cv2.imshow('preds', image_disp)
        cv2.waitKey()

def predict_video(net, path, write_video=False):
    """

    :param net:
    :param path:
    :return:
    """
    files = os.listdir(path)
    random.shuffle(files)
    print('files num:', len(files))
    for file in files:

        Video_full_path = os.path.join(path, file)
        cap = cv2.VideoCapture(Video_full_path)

        if cap.isOpened():
            ret, image = cap.read()
            if write_video:
                Video_full_out_path = os.path.join(path, 'out_' + file)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(Video_full_out_path, fourcc, 30.0, (image.shape[1], image.shape[0]), True)

            while (ret):
                ret, image = cap.read()
                if ret:
                    if image is None:
                        continue
                    # print('original image shape:', image.shape)
                    image_disp = image.copy()
                    image_origin = cv2.resize(image, (320, 160))
                    image_pre = (image_origin.astype(np.float32) / 255. - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
                    # print('resized image shape:', image_pre.shape)
                    image = np.expand_dims(image_pre, axis=0)

                    image = (image.astype(np.float32))
                    image = np.transpose(image, [0, 3, 1, 2])

                    net.eval()
                    boxes, num_boxes = net(paddle.to_tensor(image))

                    # print('preds:', num_boxes)
                    for i, pred in enumerate(boxes):
                        if pred[1] > 0.5:
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

                    if write_video:
                        out.write(image_disp)
                    cv2.imshow('preds', image_disp)
                    cv2.waitKey(1)

if __name__ == '__main__':
    net = PPYoloTiny()
    # layer_state_dict = paddle.load('./MobileNetV3_small_x1_0_ssld_pretrained.pdparams')
    # layer_state_dict = paddle.load('./model_final.pdparams')
    # layer_state_dict = paddle.load('./model_final-tiny.pdparams')
    layer_state_dict = paddle.load("/home/baidu/Documents/PaddleDetection/output/ppyolo_tiny_650e_coco/3.pdparams")
    net.set_state_dict(layer_state_dict)
    paddle.flops(net, (1, 3, 160, 320))
    # images_path = '/Users/leisheng526/Development/baidu/iov-bj/base-lane-lines-finding/hwp/' \
    #               'images/sand-storm-jianghuai-normal-03-curve_30-1487-1583'
    images_path = "/media/baidu/ssd2/ppyolo/data_mp4/images_test"
    predict_images(net, images_path)
    # predict_video(net, './video', write_video=False)
