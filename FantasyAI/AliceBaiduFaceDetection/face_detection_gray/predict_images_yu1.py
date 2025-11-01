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
            if pred[1] > 0.1: #0.75:
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

from reader import TrainReader, EvalReader
from vehicle_pdc_hand import HandDataSet

if __name__ == '__main__':
    net = PPYoloTiny()
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
        path_prune_pickle = "/root/paddlejob/workspace/env_run/afs_aicv/filelist/L1Norm_sen_yb_sense60.pickle"
        pruner.sensitive(sen_file=path_prune_pickle)
        paddle.flops(net, (1, 3, 160, 288))
        print("+1" * 100)
        plan = pruner.sensitive_prune(0.725)
        paddle.flops(net, (1, 3, 160, 288))
        print("+2" * 100)
    # for name in net.state_dict().keys():
    #     print (name, net.state_dict()[name].shape)
    #
    images_root = '/home/baidu/Documents/PaddleDetection/dataset/roadsign_voc_hand'
    train_label_file = ['/home/baidu/Documents/PaddleDetection/dataset/roadsign_voc_hand/val2.txt']
    eval_label_file = ['/home/baidu/Documents/PaddleDetection/dataset/roadsign_voc_hand/val2_200.txt']
    # model_save_path = '/media/baidu/ssd2/ppyolo/6w_data/paddle_car/paddle_car/model_hand/' + model_name_prefix
    pretrain_weight = '/home/baidu/.cache/paddle/weights/MobileNetV3_large_x0_5_pretrained'
    tag_maps = []
    tag_name = []
    tags = [6, 7, 8, 12]
    tag_name.append("non-vehicle")
    tag_maps.append(tags)
    tags = [1, 2, 3, 4]
    tag_maps.append(tags)
    tag_name.append("vehicle")
    tags = [5]
    tag_name.append("person")
    tag_maps.append(tags)
    sample_transforms = [
        {'Decode': {}},
        {'Resize': {'target_size': [160, 320], 'keep_ratio': False, 'interp': 2}},
        {'NormalizeImage': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'is_scale': True}},
        {'Permute': {}}]
    eval_dataset = HandDataSet(image_dir=images_root, anno_path=eval_label_file,
                                     tags_map=zip(tag_maps, tag_name), use_cloud=False)
    eval_loader = EvalReader(batch_size=1, sample_transforms=sample_transforms, drop_empty=True,
                             num_classes=1)
    eval_loader(eval_dataset, worker_num=4)
    from paddleslim import L1NormFilterPruner
    from paddleslim.analysis import dygraph_flops
    flops = dygraph_flops(net, [1, 3, 160, 320])
    print("FLOPs before pruning:",  flops)


