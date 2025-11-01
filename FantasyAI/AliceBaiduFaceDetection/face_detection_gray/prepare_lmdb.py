"""
读取import.txt生成对应lmdb数据
"""
import os
import json
import cv2
import numpy as np
import random
from lmdb_util import LMDB

def prepare_data(images_root, anno_path_list, lmdb_work_path):
    """
    读取import.txt生成对应lmdb数据
    :param images_root: 图片根目录
    :param anno_path_list: import.txt路径
    :param lmdb_work_path: lmdb存储路径，下有train和eval子目录
    :return:
    """
    lmdb_train_path = os.path.join(lmdb_work_path, 'train')
    lmdb_eval_path = os.path.join(lmdb_work_path, 'eval')
    if not os.path.exists(lmdb_train_path):
        os.makedirs(lmdb_train_path)
    if not os.path.exists(lmdb_eval_path):
        os.makedirs(lmdb_eval_path)
    train_db = LMDB(lmdb_train_path, 'w')
    eval_db = LMDB(lmdb_eval_path, 'w')
    ct = 0
    for anno_path in anno_path_list:
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
                if not use_cloud:
                    img_rela_path = img_rela_path.split('/')[-1]  # 本地测试使用

                img_file = os.path.join(images_root, img_rela_path)
                im_id = np.array([ct])

                if not os.path.exists(img_file):
                    print("image file not exist: %s" % img_file)
                    continue

                img = cv2.imread(img_file)
                img_h, img_w = img.shape[:2]
                wh_ratio = img_w / float(img_h)
                target_wh_ratio = target_w / float(target_h)
                if target_wh_ratio >= wh_ratio:
                    new_w = int(img_w)
                    new_h = int(img_w * target_h / target_w)
                    start_w = 0
                    start_h = int(random.uniform(0, img_h - new_h))
                else:
                    new_w = int(img_h * target_w / target_h)
                    new_h = int(img_h)
                    start_w = int(random.uniform(0, img_w - new_w))
                    start_h = 0
                img = img[start_h: start_h + new_h, start_w: start_w + new_w]
                scale = target_w / float(new_w)
                img = cv2.resize(img, (target_w, target_h))

                result = labeldata['result']

                gt_bbox = []
                gt_class = []
                gt_score = []
                for i, result_item in enumerate(result):
                    tag = int(result_item['tag'])
                    clsid = -1
                    for i, tags in enumerate(tag_maps):
                        if tag in tags:
                            cname = tag_name[i]
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

                    x1 = int((x1 - start_w) * scale)
                    x2 = int((x2 - start_w) * scale)
                    y1 = int((y1 - start_h) * scale)
                    y2 = int((y2 - start_h) * scale)

                    gt_bbox.append([x1, y1, x2, y2])
                    gt_class.append([clsid])
                    gt_score.append([1.])

                gt_bbox = np.array(gt_bbox).astype('float32')
                gt_class = np.array(gt_class).astype('int32')
                gt_score = np.array(gt_score).astype('float32')

                gt_rec = {
                    'im_id': im_id,
                    'image': img.tobytes(),
                    'h': target_h,
                    'w': target_w,
                    'gt_class': gt_class,
                    'gt_score': gt_score,
                    'gt_bbox': gt_bbox,
                }

                if len(result) != 0:
                    train_db.insert(gt_rec)
                    if ct % eval_per_sample == 0:
                        eval_db.insert(gt_rec)
                    ct += 1

    train_db.write_lmdb()
    eval_db.write_lmdb()

if __name__ == "__main__":
    use_cloud = False
    target_w = 640
    target_h = 320
    eval_per_sample = 100   # 每个多少sample选取一个eval sample

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

    if use_cloud:
        images_root = './afs_aicv/'
        anno_path_list = ['./afs/labels/ADT-perception-obstacle-train.txt']
        lmdb_work_path = '/root/data_lmdb/adt_obstacle'
        lmdb_afs_path = './afs/yuyuantuo/vehicle/data_lmdb/adt_obstacle'
    else:
        images_root = '/Users/yuyuantuo/Downloads/image_samples100'
        anno_path_list = ['/Users/yuyuantuo/Downloads/import_sample100.txt']
        lmdb_work_path = '/Users/yuyuantuo/Desktop/data_lmdb/samples100'

    prepare_data(images_root, anno_path_list, lmdb_work_path)
    # 在PaddleCloud上操作时先在work path生成数据后迁移至afs path，afs只能用于存储lmdb格式数据不能读写
    if use_cloud:
        if not os.path.exists(lmdb_afs_path):
            os.makedirs(lmdb_afs_path)
        os.system('mv -r %s %s' % (lmdb_work_path, lmdb_afs_path))
        print('finish moving lmdb.')