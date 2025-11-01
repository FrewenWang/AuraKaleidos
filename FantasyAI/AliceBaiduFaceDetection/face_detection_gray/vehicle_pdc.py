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

import os
import numpy as np
import json

from dataset import DetDataset
import paddle


# @register
# @serializable
class VehiclePDCDataSet(DetDataset):
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
        super(VehiclePDCDataSet, self).__init__(
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
                    line_strip = line.strip()
                    json_beg = line_strip.find("{")
                    json_end = line_strip.rfind("}")
                    json_item = line_strip[json_beg:(json_end + 1)]

                    try:
                        contents = json.loads(json_item)
                    except:
                        continue

                    if 'labelData' in contents:
                        labeldata = contents['labelData']  # 2015
                    elif 'layer_base' in contents:
                        labeldata = contents['layer_base']  # 2018
                    else:
                        print('error')
                        print(contents)
                        continue
                    if 'datasetsRelatedFiles' not in contents:
                        continue

                    img_rela_path = contents["datasetsRelatedFiles"][0]["bos_key"]
                    if not self.use_cloud:
                        img_rela_path = img_rela_path.split('/')[-1]  # 本地测试使用

                    img_file = os.path.join(image_dir, img_rela_path)
                    im_id = np.array([ct])

#                     if not os.path.exists(img_file):
#                         print("image file not exist")
#                         # continue

                    im_h, im_w = -1, -1
                    scale = 1.0
                    scale_w = int(im_w * scale)
                    scale_h = int(im_h * scale)
                    result = labeldata['result']

                    gt_bbox = []
                    gt_class = []
                    gt_score = []
                    # difficult = []
                    for i, result_item in enumerate(result):
                        tag = int(result_item['tag'])
                        clsid = -1
                        for i, tags in enumerate(self.tags_map):
                            if tag in tags:
                                cname = self.tags_name[i]
                                clsid = i
                                break
                            else:
                                continue
                        if clsid == -1:
                            continue
                        h = result_item['h']
                        w = result_item['w']
                        y = result_item['y']
                        x = result_item['x']
                        x1 = x
                        x2 = x + w
                        y1 = y
                        y2 = y + h

                        x1 *= scale
                        x2 *= scale
                        y1 *= scale
                        y2 *= scale
                        gt_bbox.append([x1, y1, x2, y2])
                        gt_class.append([clsid])
                        gt_score.append([1.])
                        # difficult.append([0])

                    gt_bbox = np.array(gt_bbox).astype('float32')
                    gt_class = np.array(gt_class).astype('int32')
                    gt_score = np.array(gt_score).astype('float32')

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

                    if len(result) != 0:
                        records.append(voc_rec)

                    ct += 1
                    if self.sample_num > 0 and ct >= self.sample_num:
                        break
            assert len(records) > 0, 'not found any voc record in %s' % (
                self.anno_path)
            print('{} samples in file {}'.format(ct, anno_path))
            # logger.debug('{} samples in file {}'.format(ct, anno_path))
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
    dataset = VehiclePDCDataSet(image_dir=images_root, anno_path=train_label_file,
                                tags_map=zip(tag_maps, tag_name))
    dataset.parse_dataset()

    # train_loader = paddle.io.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    # train_loader = paddle.io.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    # train_loader = paddle.io.BatchSampler(dataset, batch_size=4)
    # for batch_ind, data in enumerate(train_loader()):
    #     print(batch_ind)