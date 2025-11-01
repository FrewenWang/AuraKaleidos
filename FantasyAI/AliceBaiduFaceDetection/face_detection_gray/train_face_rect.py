"""
PaddleCloud 训练主文件，支持本地离线训练
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os, sys
import time, datetime
import train_utils
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from vehicle_pdc_facedetect import FaceDataSet
from reader import TrainReader, EvalReader

from PPYoloMobileNetV3 import PPYoloTiny
from metrics import VOCMetric

# parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
# if parent_path not in sys.path:
#     sys.path.append(parent_path)

import random
import numpy as np
# ignore warning log
import warnings
warnings.filterwarnings('ignore')

import stats as stats
from logger import setup_logger
logger = setup_logger('reader')
# import time

def print_training_status(status, log_per_step, end_epoch):
    epoch_id = status['epoch_id']
    step_id = status['step_id']
    steps_per_epoch = status['steps_per_epoch']
    training_staus = status['training_staus']
    batch_time = status['batch_time']
    data_time = status['data_time']

    logs = training_staus.log()
    space_fmt = ':' + str(len(str(steps_per_epoch))) + 'd'
    if step_id % log_per_step == 0:
        eta_steps = (end_epoch - epoch_id) * steps_per_epoch - step_id
        eta_sec = eta_steps * batch_time.global_avg
        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
        fmt = ' '.join([
            'Epoch: [{}]',
            '[{' + space_fmt + '}/{}]',
            'learning_rate: {lr:.6f}',
            '{meters}',
            'eta: {eta}',
            'batch_cost: {btime}',
            'data_cost: {dtime}',
        ])
        fmt = fmt.format(
            epoch_id,
            step_id,
            steps_per_epoch,
            lr=status['learning_rate'],
            meters=logs,
            eta=eta_str,
            btime=str(batch_time),
            dtime=str(data_time))
        logger.info(fmt)


def load_resume_weight(model, weight, optimizer=None):
    """
    继续训练中断的模型
    :param model:
    :param weight:
    :param optimizer:
    :return:
    """
    pdparam_path = weight + '.pdparams'
    if not os.path.exists(pdparam_path):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(pdparam_path))

    param_state_dict = paddle.load(pdparam_path)
    model_dict = model.state_dict()
    model_weight = {}
    incorrect_keys = 0

    for key in model_dict.keys():
        if key in param_state_dict.keys():
            model_weight[key] = param_state_dict[key]
        else:
            logger.info('Unmatched key: {}'.format(key))
            incorrect_keys += 1

    assert incorrect_keys == 0, "Load weight {} incorrectly, \
            {} keys unmatched, please check again.".format(weight,
                                                           incorrect_keys)
    logger.info('Finish resuming model weights: {}'.format(pdparam_path))

    model.set_dict(model_weight)

    last_epoch = 0
    if optimizer is not None and os.path.exists(weight + '.pdopt'):
        optim_state_dict = paddle.load(weight + '.pdopt')
        # to solve resume bug, will it be fixed in paddle 2.0
        for key in optimizer.state_dict().keys():
            if not key in optim_state_dict.keys():
                optim_state_dict[key] = optimizer.state_dict()[key]
        if 'last_epoch' in optim_state_dict:
            last_epoch = optim_state_dict.pop('last_epoch')
        optimizer.set_state_dict(optim_state_dict)

    return last_epoch


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
    param_state_dict_fuseBN = {}

    for name, weight in param_state_dict.items():
        # print('@@@@@@@######', name, weight.shape)
        if name in model_dict.keys():

            if list(weight.shape) != list(model_dict[name].shape):
                logger.info(
                    '{} not used, shape {} unmatched with {} in model.'.format(
                        name, weight.shape, list(model_dict[name].shape)))
                ignore_weights.add(name)
        else:
            logger.info('Redundant weight {} and ignore it.'.format(name))
            ignore_weights.add(name)

        #         if "backbone." + name in model_dict.keys():
        #             param_state_dict_fuseBN["backbone." + name] = param_state_dict[name]
        #         if name.replace('._layer', '') in model_dict.keys():
        #             param_state_dict_fuseBN[name.replace('._layer', '')] = param_state_dict[name]
        #             if "backbone." + name == 'backbone.conv0.weight':
        #                 param_state_dict_fuseBN["backbone." + name] = param_state_dict[name]/255
        #             else:
        #                 param_state_dict_fuseBN["backbone." + name] = param_state_dict[name]

    for weight in ignore_weights:
        param_state_dict.pop(weight, None)

    #     model.set_dict(param_state_dict_fuseBN)
    model.set_dict(param_state_dict)
    logger.info('Finish loading model weights: {}'.format(weights_path))


def eval_with_loader(loader, Model, metric):
    sample_num = 0
    Model.eval()
    for step_id, data in enumerate(loader):
        # status['step_id'] = step_id
        # 
        img = data['image']
        outputs = Model(img)

        # update metrics
        metric.update(data, outputs)

        sample_num += data['im_id'].numpy().shape[0]
        # step_id = status['step_id']
        if step_id % 100 == 0:
            logger.info("Eval iter: {}".format(step_id))
    # accumulate metric to log out
    metric.accumulate()
    metric.log()

    return sample_num


def train(train_loader, eval_loader, model, lr, optimizer, StartEpoch, EndEpoch,
          LrChangeEpoches, metric, use_gpu=True, use_VDL=False):
    """
    开始进行模型的训练
    Args:
        train_loader:    加载训练数据
        eval_loader:
        model:
        lr:
        optimizer:
        StartEpoch:     预训练模型的权重的last_epoch
        EndEpoch:
        LrChangeEpoches:
        metric:
        use_gpu:
        use_VDL:

    Returns:

    """
    place = paddle.set_device('gpu' if use_gpu else 'cpu')
    # TODO:use_ema
    # TODO:分布式训练

    steps_per_epoch = len(train_loader)
    status = {}

    # TODO callbacks
    # initial default callbacks
    # _init_callbacks()

    # TODO metrics
    if use_VDL:
        try:
            from visualdl import LogWriter
            vdl_writer = LogWriter(os.getenv("VDL_LOG_PATH"))
            vdl_loss_step = 0
            vdl_mAP_step = 0

        except Exception as e:
            print('Error:', e)

    # 设置训练日志
    status.update({
        'epoch_id': StartEpoch,
        'step_id': 0,
        'steps_per_epoch': steps_per_epoch
    })

    status['batch_time'] = stats.SmoothedValue(
        log_per_step, fmt='{avg:.4f}')
    status['data_time'] = stats.SmoothedValue(
        log_per_step, fmt='{avg:.4f}')
    status['training_staus'] = stats.TrainingStats(log_per_step)

    # =============Train=============
    for epoch_id in range(StartEpoch, EndEpoch):
        model.train()  # set to train mode
        status['mode'] = 'train'
        status['epoch_id'] = epoch_id
        train_loader.dataset.set_epoch(epoch_id)
        iter_tic = time.time()
        time_name_prefix = time.strftime('%Y%m%d%H%M%S', time.localtime(iter_tic))  # 把获取的时间转换成"年月日格式”
        if epoch_id % snapshot_epoch == 0:
            try:
                if dist.get_rank() == 0:
                    # paddle.save(model.state_dict(),
                    #             model_save_path + '/%s_epoch%d.pdparams' % (time_name_prefix, epoch_id))
                    # paddle.save(optimizer.state_dict(),
                    #             model_save_path + '/%s_epoch%d.pdopt' % (time_name_prefix, epoch_id))
                    paddle.save(model.state_dict(),
                                os.path.join(model_save_path, "epoch" + str(epoch_id) + ".pdparams"))
                    paddle.save(optimizer.state_dict(),
                                os.path.join(model_save_path, "epoch" + str(epoch_id) + ".pdopt"))
            except Exception as e:
                print('Error:', e)
            with paddle.no_grad():
                status['save_best_model'] = True
                status['mode'] = 'eval'
                tic = time.time()
                sample_num = eval_with_loader(eval_loader, model, metric)
                status['sample_num'] = sample_num
                status['cost_time'] = time.time() - tic
                if use_VDL:  # vdl mAP曲线
                    for key, map_value in metric.get_results().items():
                        try:
                            vdl_writer.add_scalar("{}-mAP".format(key), map_value[0], vdl_mAP_step)
                        except Exception as e:
                            print('Error:', e)

                    vdl_mAP_step += 1
                # reset metric states for metric may performed multiple times
                metric.reset()
        # try:
        for step_id, data in enumerate(train_loader):
            if step_id % 1000 == 0 and step_id > 0:
                try:
                    if dist.get_rank() == 0:
                        # paddle.save(model.state_dict(),
                        #             model_save_path + '/%s_epoch%d_%d.pdparams' % (time_name_prefix, epoch_id, step_id))
                        # paddle.save(optimizer.state_dict(),
                        #             model_save_path + '/%s_epoch%d_%d.pdopt' % (time_name_prefix, epoch_id, step_id))
                        paddle.save(model.state_dict(),
                                    os.path.join(model_save_path, "epoch" + str(epoch_id) + ".pdparams"))
                        paddle.save(optimizer.state_dict(),
                                    os.path.join(model_save_path, "epoch" + str(epoch_id) + ".pdopt"))

                    # pathsave1 = "/root/paddlejob/workspace/log/models"
                    # if not os.path.isdir(pathsave1):
                    #     os.makedirs(pathsave1)
                    # paddle.save(model.state_dict(),
                    #             pathsave1 + '/%s_epoch%d_%d.pdparams' % (time_name_prefix, epoch_id, step_id))
                    # paddle.save(optimizer.state_dict(),
                    #             pathsave1 + '/%s_epoch%d_%d.pdopt' % (time_name_prefix, epoch_id, step_id))

                except Exception as e:
                    print('Error:', e)

            status['data_time'].update(time.time() - iter_tic)
            status['step_id'] = step_id
            img = data['image']
            model.train()
            outputs = model(img, data)
            loss = outputs['loss']
            # model backward
            loss.backward()    # 反向传播
            optimizer.step()

            curr_lr = optimizer.get_lr()
            optimizer.clear_grad()

            status['learning_rate'] = curr_lr
            status['training_staus'].update(outputs)
            status['batch_time'].update(time.time() - iter_tic)

            if step_id % log_per_step == 0:
                print_training_status(status, log_per_step, EndEpoch)
                if use_VDL:  # vdl loss曲线
                    try:
                        for loss_name, loss_value in status['training_staus'].get().items():
                            vdl_writer.add_scalar(loss_name, loss_value, vdl_loss_step)
                            vdl_loss_step += 1

                    except Exception as e:
                        print('Error:', e)
            iter_tic = time.time()
        lr.step()

# 开始执行模型训练，默认使用云端训练
if __name__ == "__main__":
    use_cloud = False
    usePruner = True     # False 是否使用剪枝后的模型 TODO：什么是模型剪枝？？？
    useQuanter = False   # 是否训练量化模型
    worker_scale = 1
    use_gpu = True      # 是否使用GPU
    useMutilpleGPU = True   # 是否使用多核GPU

    useVDL = False  # paddle的深度学习可视化框架，类似于Tensorflow2的里面的Tensorboard

    time_name = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))  # 把获取的时间转换成"年月日格式”
    if useMutilpleGPU:
        # 初始化动态图模式下的并行训练环境。 用于在单机多卡的情况下。执行多卡的并行训练
        dist.init_parallel_env()
    # 使用当前的时间戳作为模型的前缀
    model_name_prefix = str(time_name)  # 线上模型文件夹和模型名字都用该前缀
    if use_cloud:
        # os.c7230302system('sh ./each_node.sh')  # 挂载多个afs
        images_root = '/root/paddlejob/data_train'
        #train_label_file = ['./afs/labels/ADT-perception-obstacle-train.txt']
        # train_label_file = ['./afs/labels/ADT-perception-obstacle-train.txt',
        #                     './afs/labels/L3-perception-data-wheel.txt']
        # eval_label_file = ['./afs/labels/ADT-perception-obstacle-eval.txt']
        # 训练数据的标签数据
        train_label_file = [
            '/home/baiduiov/Pictures/face_rect_dataset/filelist/train_filelist/trainV2.txt',
            '/home/baiduiov/Pictures/face_rect_dataset/filelist/filelist/file_omsface_biaozhu1.txt',
            '/root/paddlejob/data_train/filelist/train_filelist/oms_SmallFace_biaozhu_20211118.txt',
            # widerface数据476张，openImage3721张， 4200张
            '/home/baiduiov/Pictures/face_rect_dataset/filelist/train_filelist/widerMoreFace_476.txt',
            '/home/baiduiov/Pictures/face_rect_dataset/filelist/openImageMoreface3700.txt',
            # widerface数据476张，openImage3721张， 4200张
            '/home/baiduiov/Pictures/face_rect_dataset/filelist/train_filelist/widerMoreFace_476.txt',
            '/home/baiduiov/Pictures/face_rect_dataset/filelist/train_filelist/openImageMoreface3700.txt',
            '/home/baiduiov/Pictures/face_rect_dataset/filelist/train_filelist/jidu_data.txt',
            '/home/baiduiov/Pictures/face_rect_dataset/filelist/train_filelist/jidu_data_1107.txt',
            '/home/baiduiov/Pictures/face_rect_dataset/filelist/train_filelist/jidu_data_1108.txt',
            '/home/baiduiov/Pictures/face_rect_dataset/filelist/train_filelist/jidu_data_0112.txt',
            '/home/baiduiov/Pictures/face_rect_dataset/filelist/train_filelist/jidu_data_0227.txt',
            '/home/baiduiov/Pictures/face_rect_dataset/filelist/train_filelist/jidu_data_0227_fl.txt',
        ]

        eval_label_file = ['/home/baiduiov/Pictures/face_rect_dataset/filelist/val_filelist/val1.txt',
                           '/home/baiduiov/Pictures/face_rect_dataset/filelist/val_filelist/val_omsface1.txt']
        # 模型剪枝的标签数据
        eval_label_file_prune = ['/home/baiduiov/Pictures/face_rect_dataset/filelist/val200.txt']
        model_save_path = ('/home/baiduiov/03.ProgramSpace/20.AI/01.WorkSpace/AliceAILearning/AliceBaiduFaceDetection'
                           '/face_detection_gray/output/')    # + model_name_prefix

        # pretrain_weight = './afs/leisheng526/vehicle/pretrain_models/model_final-tiny'
        #  pretrain_weight = "./afs_aicv/filelist/2d4f5904_noPrune_noQuant_pd222"
        # pretrain_weight = "./afs_aicv/filelist/dan_2d4f_yuanshi"
        # 预训练权重模型数据
        pretrain_weight = "./afs_aicv/filelist/1107V4_noFuse_tmp20230313"
        pretrain_weight = "./afs_aicv/filelist/addhaze_20230418_nofusev2"
        # 设置预训练权重为空。代表不使用预训练
        pretrain_weight = ""
        path_prune_pickle = "./afs_aicv/filelist/dan_2d4f_prune.pickle"
        # TODO all_back_from_ImgNet是个什么东西
        pathNeg = "/root/paddlejob/data_train/all_back_from_ImgNet"
        # 带口罩数据
        pathKouzhao = "/root/paddlejob/data_train/kouzhao_mult"
        resume_weight = ''
    else:
        # 图片的根路径
        images_root = '/home/baiduiov/Pictures/face_rect_dataset/'
        # 训练数据的标签文件
        train_label_file = ["/home/baiduiov/Pictures/face_rect_dataset/filelist/val200.txt"]
        # 校验数据的标签剪枝数据
        eval_label_file_prune = ["/home/baiduiov/Pictures/face_rect_dataset/filelist/val200.txt"]
        eval_label_file = ["/home/baiduiov/Pictures/face_rect_dataset/filelist/val200.txt"]

        # 模型存储的路径
        model_save_path = (f'/home/baiduiov/03.ProgramSpace/20.AI/01.WorkSpace/AliceAILearning/AliceBaiduFaceDetection'
                           '/face_detection_gray/output/')
        # 使用预训练权重文件
        pretrain_weight = '/home/yubin/.cache/paddle/weights/MobileNetV3_large_x0_5_pretrained'
        pretrain_weight = "/media/yubin/3.6TB_SSD/facedetect/缩小尺寸/models_save/剪枝模型/v1/20211111103031_epoch24_2000"
        pretrain_weight = "/media/yubin/3.6TB_SSD/facedetect/缩小尺寸/models_save/芯算一体/c7230302/pretrained/c7230302_weights"
        pretrain_weight = "/media/yubin/3.6TB_SSD/facedetect/缩小尺寸/models_save/芯算一体/c7230302/epoch19"
        pretrain_weight = '/media/yubin/3.6TB_SSD/facedetect/缩小尺寸/models_save/芯算一体/2d4f5904/epoch19'
        pretrain_weight = ""
        # 使用剪枝文件
        path_prune_pickle = "/home/baiduiov/03.ProgramSpace/20.AI/01.WorkSpace/AliceAILearning/AliceBaiduFaceDetection/face_detection_gray/assets/dan_2d4f_prune.pickle"
        pathNeg = "/media/yubin/3.6TB_SSD/val_data/back_noHand_img/back_from_ImgNet"
        resume_weight = ''

    try:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
    except Exception as e:
        print('Error:', e)

    # 启动轮次,和结束的轮次，我们共计训练20轮
    start_epoch = 0
    end_epoch = 20
    lr_change_epoches = [8, 18, 23]
    # 每隔10次打印一次额训练日志
    log_per_step = 10
    snapshot_epoch = 1

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
    num_classes = len(tag_name)

    # 构建YOLO输出的结果信息
    tag_maps = [[0]]
    tag_name = ["facebox"]
    # 分类数量
    num_classes = 1

    # 构建训练数据集
    dataset = FaceDataSet(image_dir=images_root, anno_path=train_label_file,
                                tags_map=zip(tag_maps, tag_name), use_cloud=use_cloud, is_train=True)
    # 剪枝量化的数据
    eval_dataset_prune = FaceDataSet(image_dir=images_root, anno_path=eval_label_file_prune,
                                     tags_map=zip(tag_maps, tag_name), use_cloud=use_cloud, is_train=False)
    # 测试数据集
    eval_dataset = FaceDataSet(image_dir=images_root, anno_path=eval_label_file,
                                     tags_map=zip(tag_maps, tag_name), use_cloud=use_cloud, is_train=False)

    # 这个文件集合是什么东西？？？？
    mix_back_imgnet2 = train_utils.getAllFiles(pathNeg)
    pathInfos = []
    #

    for path_train_item in train_label_file:
        pathInfos += open(path_train_item).readlines()

    allNegFiles = copy.deepcopy(mix_back_imgnet2)

    # 口罩相关的数据
    imgkouzhaos = train_utils.getAllFiles(pathKouzhao)

    # 进行训练样本数据转换
    sample_transforms = [
        {'Decode': {}},
        {"RandomRotate": {"prob": 0.5}},
        {'Mixup_MergeNewFace': {"pathInfos": pathInfos, "pathRoot": images_root, "prob": 0.2, "pathNegs": allNegFiles}},
        {'Mixup_Imgnet': {"back_imgs": mix_back_imgnet2, "prob": 0.5}},

        {"RandomBlurNoise": {"prob": 0.5}},
        {'RandomDistort': {}},
        # {'RandomExpand': {'fill_value': [123.675, 116.28, 103.53], 'prob':0.3, 'ratio':2} },
        {'RandomExpand': {'fill_value': [127.5], 'prob': 0.3, 'ratio': 2}},
        # {'RandomCrop': {}},
        # {'RandomFlip_Ver':{'prob':0.6}},
        {'RandomFlip': {'prob': 0.5, "use_cloud": True, "imgKouZhaos": imgkouzhaos}}
    ]
    batch_transforms = [
        {'BatchRandomResize':
             {'target_size': [[192, 288], [224, 320], [160, 288], [192, 352]],
              'random_size': True, 'random_interp': True, 'keep_ratio': False}},
        {'NormalizeBox': {}},
        {'PadBox': {'num_max_boxes': 100}}, {'BboxXYXY2XYWH': {}},
        {'NormalizeImage': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'is_scale': True}},
        {'Permute': {}},
        {'Gt2YoloTarget':
             {'anchor_masks': [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
              'anchors': [[10, 15], [24, 36], [72, 42],
                          [35, 87], [102, 96], [60, 170],
                          [220, 125], [128, 222], [264, 266]],
              'downsample_ratios': [32, 16, 8]}}]

    #
    train_loader = TrainReader(batch_size=32, sample_transforms=sample_transforms, batch_transforms=batch_transforms,
                               shuffle=True, drop_last=True, use_shared_memory=True, num_classes=num_classes)
    train_loader(dataset, worker_num=(8 * worker_scale))

    sample_transforms = [
        {'Decode': {}},
        {'Resize': {'target_size': [160, 288], 'keep_ratio': False, 'interp': 2}},
        {'NormalizeImage': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225], 'is_scale': True}},
        {'Permute': {}}]
    # eval_batch_sampler = paddle.io.BatchSampler(eval_dataset, batch_size=1)
    eval_loader_prune = EvalReader(batch_size=1, sample_transforms=sample_transforms, drop_empty=True,
                             num_classes=num_classes)
    eval_loader_prune(eval_dataset_prune, worker_num=1)

    eval_loader = EvalReader(batch_size=1, sample_transforms=sample_transforms, drop_empty=True,
                                   num_classes=num_classes)
    eval_loader(eval_dataset, worker_num=1)

    # 实例化PPYoloTiny对象
    model = PPYoloTiny(model="2d4f5904")
    paddle.flops(model, (1, 1, 160, 288))
    print("**********************************************************")
    # 该接口提供一种学习率按 指定轮数 进行衰减的策略。
    # 文章参考：https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/lr/MultiStepDecay_cn.html#multistepdecay
    # 主要是进行动态调整学习率
    lr = paddle.optimizer.lr.MultiStepDecay(learning_rate=0.001, milestones=[7, 14, 17], gamma=0.1)
    lr = paddle.optimizer.lr.MultiStepDecay(learning_rate=0.0001, milestones=[15, 18, ], gamma=0.1)

    optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters(), weight_decay=1e-4)
    metric =VOCMetric(label_list=tag_name, map_type='integral', class_num=num_classes)
    metric.reset()

    # 使用paddle做剪枝量化的逻辑
    if usePruner:
        try:
            from paddleslim.dygraph import L1NormFilterPruner
        except:
            os.system('pip3 install paddleslim==2.1.1')
            from paddleslim.dygraph import L1NormFilterPruner
#         from paddleslim.dygraph import L1NormFilterPruner
        pruner = L1NormFilterPruner(model, [1, 1, 160, 288])
        def eval_fn():
            eval_with_loader(eval_loader_prune, model, metric)
            value = metric.detection_map.mAP
            metric.reset()
            return value
        # sen = pruner.sensitive(eval_func=eval_fn, sen_file=model_save_path + "L1Norm_sen_yb_sense2.pickle") #, skip_vars=['conv2d_80.w_0','conv2d_80.w_0','conv2d_80.w_0'])
#         pruner.sensitive(sen_file=path_prune_pickle)
        pruner.sensitive(eval_func=eval_fn, sen_file=path_prune_pickle)

        paddle.flops(model, (1, 1, 160, 288))
        print ("+1"*100)
        plan = pruner.sensitive_prune(0.5)
        paddle.flops(model, (1, 1, 160, 288))
        print("+2" * 100)

    # 加载预训练模型
    if pretrain_weight:
        load_pretrain_weight(model, pretrain_weight)
        logger.info("Load weights {} to start training".format(pretrain_weight))
    elif resume_weight:
        start_epoch = load_resume_weight(model, resume_weight, optimizer)
        logger.info("Resume weights of epoch {}".format(start_epoch))
    else:
        usePruner = False

    # 判断是否使用前训练量化
    if useQuanter:
        quant_config = {
            "weight_preprocess_type": None,
            "activation_preprocess_type": None,
            "weight_quantize_type": "abs_max",
            "activation_quantize_type": "moving_average_abs_max",
            "weight_bits": 8,
            "activation_bits": 8,
            "dtype": "int8",
            "window_size": 10000,
            "moving_rate": 0.9,
            "quantizable_layer_type": ['Conv2D', 'Linear'],
        }
        import paddleslim

        quanter = paddleslim.QAT(config=quant_config)
        quanter.quantize(model)
        paddle.flops(model, (1, 1, 160, 288))
        print("+3" * 100)

    if useMutilpleGPU:
        model = paddle.DataParallel(model)
    # 进行模型的训练
    train(train_loader, eval_loader, model, lr, optimizer, start_epoch, end_epoch,
          lr_change_epoches, metric, use_gpu=use_gpu, use_VDL=useVDL)
