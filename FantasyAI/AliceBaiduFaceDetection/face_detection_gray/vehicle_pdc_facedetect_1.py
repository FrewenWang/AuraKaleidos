"""
自定义数据集，支持aicv 前车行人非机动车数据import.txt格式
"""
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import os
import random
import cv2
import numpy as np
import json
from dataset import DetDataset
import paddle

import xml.etree.ElementTree as et
import cv2 as cv
def get_label_box(pathxml):
    # path = "/media/baidu/ssd2/ppyolo/6w_data/baidu_hand_xml/gesture/fist/12_00010.xml"
    info = et.parse(pathxml)
    root = info.getroot()
    obj = root.find("object")
    bndbox = obj.find("bndbox")
    xmin = bndbox.find("xmin")
    xmax = bndbox.find("xmax")
    ymin = bndbox.find("ymin")
    ymax = bndbox.find("ymax")

    l,t,r,b = map(float, [xmin.text, ymin.text, xmax.text, ymax.text])
    return [l,t,r,b]

def get_box_point_txt(pathtxt):
    allrslt = []
    infos = open(pathtxt).readlines()
    for linet in infos:
        linet = linet.strip().split("\t")
        linet1 = list(map(float, linet))
        allrslt.append(linet1)
    return allrslt

def get_box_point_xml(pathxml):
    allfiles = []
    infos = json.loads(open(pathxml).read())
    WorkLoad = infos['annotation']
    fscale = WorkLoad["object"]
    for id in range(len(fscale)):
        xmin = fscale[id]['bndbox']['xmin']
        ymin = fscale[id]['bndbox']['ymin']
        xmax = fscale[id]['bndbox']['xmax']
        ymax = fscale[id]['bndbox']['ymax']
        xmin = int(float(xmin))
        ymin = int(float(ymin))
        xmax = int(float(xmax))
        ymax = int(float(ymax))
        # l,t,r,b = map(float, [xmin.text, ymin.text, xmax.text, ymax.text])
        allfiles.append(xmin)
        allfiles.append(ymin)
        allfiles.append(xmax)
        allfiles.append(ymax)

    return allfiles

def get_box_point(pathJson):

    # label_infos = {}
    # label_infos['facebox'] = []
    # label_infos['facepoints'] = []

    try:
        infos = json.loads(open(pathJson).read())
        WorkLoad = infos['WorkLoad']
        fscale = WorkLoad["scale_x"]
        DataList = infos["DataList"]
        Point_num = WorkLoad['Point Num']
        # 目前Point_num 有三个数值，2, 106, 72,
        # 暂时先设置一个阈值5,大于等于5则使用人脸点推理方框，否则直接读取face_bbox

        if Point_num >= 5:
            all_x = []
            all_y = []
            for item in DataList:
                if item['type'] != "Point":
                    continue
                fx, fy = item['coordinates']
                # label_infos['facepoints'].append([fx, fy])
                all_x.append(fx)
                all_y.append(fy)
            left = min(all_x)
            right = max(all_x)
            top = min(all_y)
            bottom = max(all_y)
        else:
            for item in DataList:
                if item['type'] == "face_bbox":
                    coordinates = item['coordinates']
                    left = coordinates[0]['left']
                    top = coordinates[1]['top']
                    right = coordinates[2]['right']
                    bottom = coordinates[3]['bottom']
                    # label_infos['facebox'] = [left, top, right, bottom]
        if right-left < 10 or bottom - top < 10:
            return []
        return [left*fscale, top*fscale, right*fscale, bottom*fscale]
    except:
        # print ("pathJson error:", pathJson)
        return []

    # return label_infos

# @register
# @serializable
class FaceDataSet(DetDataset):
    """
    Load dataset with PascalVOC format.

    Notes:
    `anno_path` must contains xml file and image file path for annotations.

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): voc annotation file path.
        data_fields (list): key name of data dictionary, at least have 'image'.
        sample_num (int): number of samples to load, -1 means all.
        label_list (str): if use_default_label is False, will load
            mapping between category and class index.
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 sample_num=-1,
                 label_list=None,
                 tags_map=None,
                 use_cloud=False):
        super(FaceDataSet, self).__init__(
            dataset_dir=dataset_dir,
            image_dir=image_dir,
            anno_path=anno_path,
            data_fields=data_fields,
            sample_num=sample_num)
        self.tags_map, self.tags_name = zip(*tags_map)
        self.label_list = label_list
        self.use_cloud = use_cloud
        # self.parse_dataset()


    def parse_dataset(self, ):
        """
        从本地import.txt读取标注和远端aicv的图片
        :return:
        """
        # anno_path = self.anno_path
        image_dir = self.image_dir
        records = []
        ct = 0
        for i, anno_path in enumerate(self.anno_path):
            with open(anno_path, 'r') as fr:
                while True:
                    line = fr.readline()
                    if not line:
                        break
                    pathimg1 = line.strip()
                    #根据img生成json，二者在同一路径下，只是文件后缀不一样
                    filepathInfo = pathimg1.split("/")
                    imgname = filepathInfo[-1]
                    imgname1 = imgname.split(".")[:-1]
                    imgname2 = ".".join(imgname1)
                    filepathPrefix = "/".join(filepathInfo[:-1])
                    # 该路径下没有人脸点标注信息
                    if "/1000personFacialLandmark/标2_ModifyTop/" in pathimg1 or "oms_SmallFace_biaozhu" in pathimg1:
                        jsonname = imgname2 + ".json"
                    elif "/widerMoreFace/" in pathimg1 or  "/openImageMoreface/" in pathimg1:
                        jsonname = imgname2 + ".jpg.txt"
                    
                    elif "/jidu_data/" in pathimg1:
                        jsonname = imgname2 + ".xml"
                        print('jsonname', jsonname)

                    else:
                        # 有人脸点标注信息
                        jsonname = "out_" + imgname2 + ".json"

                    # pathimg1, pathxml1 = line.strip().split("\t")
                    pathimg = os.path.join(self.image_dir, pathimg1)
                    try:
                        img = cv.imread(pathimg, 0)
                    except:
                        continue
                    pathjson1 = os.path.join(filepathPrefix, jsonname)
                    pathjson = os.path.join(self.image_dir, pathjson1)
                   # print ("path_pathjson_", pathjson)
                    img_file = pathimg
                    im_id = np.array([ct])

                    im_h, im_w = img.shape
                    scale = 1.0
                    scale_w = int(im_w * scale)
                    scale_h = int(im_h * scale)
                    # result = labeldata['result']

                    gt_bbox = []
                    gt_class = []
                    gt_score = []
                    if ".jpg.txt" in jsonname:
                        box_label = get_box_point_txt(pathjson)
                        if len(box_label) == 0:
                            continue
                        for boxitem in box_label:
                            gt_bbox.append(boxitem)                            
                            gt_class.append([0])
                            gt_score.append([1.])
                    
                    elif ".xml" in jsonname:
                        print('****jsonname', jsonname)
                        pathjson1 = os.path.join(filepathPrefix, jsonname)
                        pathjson = os.path.join(self.image_dir, pathjson1)
                        pathjson = pathjson.replace('./', '') 
                        print('****pathjson', pathjson)
                        box_label = get_box_point_xml(pathjson)
                    
                    else:
                        # x1,y1,x2,y2 = get_label_box(pathxml)
                        box_label = get_box_point(pathjson)
                        if len(box_label) == 0:
                            continue
                        # if len(labelinfos['facebox']) > 0 and len(labelinfos['facepoints'])==0:
                        #     x1,x2,y1,y2 = labelinfos['facebox']
                        #     x1 *= scale
                        #     x2 *= scale
                        #     y1 *= scale
                        #     y2 *= scale
                        #     gt_bbox.append([x1, y1, x2, y2])

                        gt_bbox.append(box_label)
                        gt_class.append([0])
                        gt_score.append([1.])
                        # difficult.append([0])

                    gt_bbox = np.array(gt_bbox).astype(np.float32)
                    gt_class = np.array(gt_class).astype(np.float32)
                    gt_score = np.array(gt_score).astype(np.float32)

                    voc_rec = {
                        'im_file': img_file,
                        'im_id': im_id,
                        'h': im_h,
                        'w': im_w
                    } if 'image' in self.data_fields else {}

                    gt_rec = {
                        'gt_class': gt_class,
                        'gt_score': gt_score,
                        'gt_bbox': gt_bbox,
                    }
                    for k, v in gt_rec.items():
                        # if k in self.data_fields:
                        voc_rec[k] = v

                    voc_rec["only_negimg"] = 0
                    voc_rec["crop_MaxHand"] = 0

                    records.append(voc_rec)

                    ct += 1
                    if self.sample_num > 0 and ct >= self.sample_num:
                        break
            assert len(records) > 0, 'not found any voc record in %s' % (
                self.anno_path)
            print('{} samples in file {}'.format(ct, anno_path))
            # logger.debug('{} samples in file {}'.format(ct, anno_path))
        # print ("records",records[:10])
        # input()

        # record_dst = []
        # record_num = len(records)
        # tmp_im_file = []
        # tmp_gt_bbox = []
        # records1 = copy.deepcopy(records)
        # for record_item in records1:
        #     for i in range(4):
        #         randId = random.randint(0, record_num-1)
        #         rand_sample = records[randId]
        #         copy_sample = copy.deepcopy(rand_sample)
        #         im_file1 = copy_sample['im_file']
        #         gt_bbox1 = copy_sample['gt_bbox']
        #         tmp_im_file.append(copy.deepcopy(im_file1))
        #         tmp_gt_bbox.append(copy.deepcopy(gt_bbox1))
        #     image1 = cv2.imread(tmp_im_file[0])
        #     image1 = np.array(image1[:,:,::-1]).astype(np.float64)
        #     record_item['image1'] = image1
        #     record_item['gt_bbox1'] = np.array(tmp_gt_bbox[0]).astype(np.float64)
            # record_item['im_file2'] = tmp_im_file[1]
            # record_item['gt_bbox2'] = tmp_gt_bbox[1]
            # record_item['im_file3'] = tmp_im_file[2]
            # record_item['gt_bbox3'] = tmp_gt_bbox[2]
            # record_item['im_file4'] = tmp_im_file[3]
            # record_item['gt_bbox4'] = tmp_gt_bbox[3]
            # record_dst.append(record_item)
            # print("item", record_item)
        self.roidbs= records

    def get_label_list(self):
        """
        类别标签文件
        :return:
        """
        return os.path.join(self.dataset_dir, self.label_list)


if __name__ == "__main__":
    images_root = '/Users/leisheng526/Development/baidu/iov-anp/PaddleENV/paddleenv/PPYolo/' \
                  'PPYOLOMobileNetV3/sampleData/image_samples100'
    train_label_file = '/Users/leisheng526/Development/baidu/iov-anp/PaddleENV/paddleenv/' \
                       'PPYolo/PPYOLOMobileNetV3/sampleData/import_sample100.txt'

    images_root = '/media/baidu/3.6TB_SSD/facedetect/原始数据'
    train_label_file = ['/media/baidu/3.6TB_SSD/facedetect/原始数据/filelist/train.txt']
    eval_label_file = ['/media/baidu/3.6TB_SSD/facedetect/原始数据/filelist/val6k.txt']
    tag_maps = []
    tag_name = []
    tags = [1, 2, 3, 4]
    tag_maps.append(tags)
    tag_name.append("vehicle")
    tags = [5]
    tag_name.append("person")
    tag_maps.append(tags)
    tags = [6, 7, 8, 12]
    tag_name.append("non-vehicle")
    tag_maps.append(tags)
    dataset = FaceDataSet(image_dir=images_root, anno_path=eval_label_file,
                                tags_map=zip(tag_maps, tag_name))
    dataset.parse_dataset()

    # train_loader = paddle.io.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    # train_loader = paddle.io.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    # train_loader = paddle.io.BatchSampler(dataset, batch_size=4)
    # for batch_ind, data in enumerate(train_loader()):
    #     print(batch_ind)
