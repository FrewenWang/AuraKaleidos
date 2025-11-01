"""
数据增强算子
15%  1张人脸
拼接更多脸，当脸的面积小于图片的1/9时，增加1或者2个脸 45%
找一张无人脸图片，随机位置出现4,5张人脸 20%
neg图片 20%，（1,人脸贴方框， 2,imgNet无人脸， 3,coco无人脸）
"""
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

# function:
#    operators to process sample,
#    eg: decode/resize/crop image

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

from numbers import Number, Integral

import uuid
import random
import math
import numpy as np
import os
import copy
import time
import operators_MergeNew
import cv2
from PIL import Image, ImageEnhance, ImageDraw
from scipy import ndimage
import json
import cv2 as cv

import bbox_utils
from op_helper import (satisfy_sample_constraint, filter_and_process,
                       generate_sample_bbox, clip_bbox, data_anchor_sampling,
                       satisfy_sample_constraint_coverage, crop_image_sampling,
                       generate_sample_bbox_square, bbox_area_sampling,
                       is_poly, gaussian_radius, draw_gaussian,
                       jaccard_overlap, gaussian2D)

from logger import setup_logger

logger = setup_logger(__name__)


class BboxError(ValueError):
    """
    Bbox error exception
    """
    pass


class ImageError(ValueError):
    """
    Image error exception
    """
    pass


class BaseOperator(object):
    """
    operator base
    """

    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def apply(self, sample, context=None):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        return sample

    def __call__(self, sample, context=None):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        if isinstance(sample, Sequence):
            for i in range(len(sample)):
                sample[i] = self.apply(sample[i], context)
        else:
            sample = self.apply(sample, context)
        return sample

    def __str__(self):
        return str(self._id)


# @register_op
class Decode(BaseOperator):
    """
    Transform the image data to numpy format following the rgb format
    """

    def __init__(self):
        """ Transform the image data to numpy format following the rgb format
        """
        super(Decode, self).__init__()

    def apply(self, sample, context=None):
        """ load image if 'im_file' field is not empty but 'image' is"""
        try:
            if 'image' not in sample:
                with open(sample['im_file'], 'rb') as f:
                    sample['image'] = f.read()
                sample.pop('im_file')
            im = sample['image']
            data = np.frombuffer(im, dtype='uint8')
            im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            sample['image'] = im
            if 'h' not in sample:
                sample['h'] = im.shape[0]
            elif sample['h'] != im.shape[0]:
                # logger.warn(
                #     "The actual image height: {} is not equal to the "
                #     "height: {} in annotation, and update sample['h'] by actual "
                #     "image height.".format(im.shape[0], sample['h']))
                sample['h'] = im.shape[0]
            if 'w' not in sample:
                sample['w'] = im.shape[1]
            elif sample['w'] != im.shape[1]:
                # logger.warn(
                #     "The actual image width: {} is not equal to the "
                #     "width: {} in annotation, and update sample['w'] by actual "
                #     "image width.".format(im.shape[1], sample['w']))
                sample['w'] = im.shape[1]
            sample['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
            sample['scale_factor'] = np.array([1., 1.], dtype=np.float32)
        except Exception:
            if 'im_file' in sample:
                sample.pop('im_file')
            randomByteArray = bytearray(os.urandom(6220800))  # 把数组赋值给OpenCV类型矩阵
            flatNumpyArray = np.array(randomByteArray)  # 矩阵变维, 1维变维2维(灰度), 1维变为3维(彩色)
            im = flatNumpyArray.reshape(1080, 1920, 3)  # cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            sample['image'] = im
            if 'h' not in sample:
                sample['h'] = im.shape[0]
            elif sample['h'] != im.shape[0]:
                # logger.warn(
                #     "The actual image height: {} is not equal to the "
                #     "height: {} in annotation, and update sample['h'] by actual "
                #     "image height.".format(im.shape[0], sample['h']))
                sample['h'] = im.shape[0]
            if 'w' not in sample:
                sample['w'] = im.shape[1]
            elif sample['w'] != im.shape[1]:
                # logger.warn(
                #     "The actual image width: {} is not equal to the "
                #     "width: {} in annotation, and update sample['w'] by actual "
                #     "image width.".format(im.shape[1], sample['w']))
                sample['w'] = im.shape[1]

            sample['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
            sample['scale_factor'] = np.array([1., 1.], dtype=np.float32)

            sample['gt_bbox'] = np.array([]).astype(np.float32)
            sample['gt_class'] = np.array([]).astype(np.float32)
            sample['gt_score'] = np.array([]).astype(np.float32)

            sample['gt_facepoints'] = np.array([]).astype(np.float32)
            sample['only_neg'] = np.array([]).astype(np.int8)

            return sample
        return sample


# @register_op
class Permute(BaseOperator):
    """
    Change the channel to be (C, H, W)
    """

    def __init__(self):
        """
        Change the channel to be (C, H, W)
        """
        super(Permute, self).__init__()

    def apply(self, sample, context=None):
        im = sample['image']
        img = np.zeros(shape=[1, im.shape[0], im.shape[1]], dtype=np.float32)
        img[0, :, :] = im
        # im = im.transpose((2, 0, 1))
        sample['image'] = img
        return sample


# @register_op
class Lighting(BaseOperator):
    """
    Lighting the image by eigenvalues and eigenvectors
    Args:
        eigval (list): eigenvalues
        eigvec (list): eigenvectors
        alphastd (float): random weight of lighting, 0.1 by default
    """

    def __init__(self, eigval, eigvec, alphastd=0.1):
        super(Lighting, self).__init__()
        self.alphastd = alphastd
        self.eigval = np.array(eigval).astype('float32')
        self.eigvec = np.array(eigvec).astype('float32')

    def apply(self, sample, context=None):
        alpha = np.random.normal(scale=self.alphastd, size=(3,))
        sample['image'] += np.dot(self.eigvec, self.eigval * alpha)
        return sample


# @register_op
class RandomErasingImage(BaseOperator):
    """
    Random Erasing Data Augmentation, see https://arxiv.org/abs/1708.04896
    """

    def __init__(self, prob=0.5, lower=0.02, higher=0.4, aspect_ratio=0.3):
        """
        Random Erasing Data Augmentation, see https://arxiv.org/abs/1708.04896
        Args:
            prob (float): probability to carry out random erasing
            lower (float): lower limit of the erasing area ratio
            heigher (float): upper limit of the erasing area ratio
            aspect_ratio (float): aspect ratio of the erasing region
        """
        super(RandomErasingImage, self).__init__()
        self.prob = prob
        self.lower = lower
        self.higher = higher
        self.aspect_ratio = aspect_ratio

    def apply(self, sample):
        gt_bbox = sample['gt_bbox']
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image is not a numpy array.".format(self))
        if len(im.shape) != 3:
            raise ImageError("{}: image is not 3-dimensional.".format(self))

        for idx in range(gt_bbox.shape[0]):
            if self.prob <= np.random.rand():
                continue

            x1, y1, x2, y2 = gt_bbox[idx, :]
            w_bbox = x2 - x1
            h_bbox = y2 - y1
            area = w_bbox * h_bbox

            target_area = random.uniform(self.lower, self.higher) * area
            aspect_ratio = random.uniform(self.aspect_ratio,
                                          1 / self.aspect_ratio)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < w_bbox and h < h_bbox:
                off_y1 = random.randint(0, int(h_bbox - h))
                off_x1 = random.randint(0, int(w_bbox - w))
                im[int(y1 + off_y1):int(y1 + off_y1 + h), int(x1 + off_x1):int(
                    x1 + off_x1 + w), :] = 0
        sample['image'] = im
        return sample


# @register_op
class NormalizeImage(BaseOperator):
    """
    Image normalization
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[1, 1, 1],
                 is_scale=True):
        """
        Args:
            mean (list): the pixel mean
            std (list): the pixel variance
        """
        super(NormalizeImage, self).__init__()
        self.mean = [0.5] #mean
        self.std = [1] #std
        self.is_scale = is_scale
        if not (isinstance(self.mean, list) and isinstance(self.std, list) and
                isinstance(self.is_scale, bool)):
            raise TypeError("{}: input type is invalid.".format(self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def apply(self, sample, context=None):
        """Normalize the image.
        Operators:
            1.(optional) Scale the image to [0,1]
            2. Each pixel minus mean and is divided by std
        """
        im = sample['image']
        im = im.astype(np.float32, copy=False)
        mean = np.array(self.mean)#[np.newaxis, np.newaxis, :]
        std = np.array(self.std)#[np.newaxis, np.newaxis, :]

        if self.is_scale:
            im = im / 255.0
#             im = im

        # im -= mean
        # im /= std

        sample['image'] = im
        return sample


# @register_op
# class GridMask(BaseOperator):
#     def __init__(self,
#                  use_h=True,
#                  use_w=True,
#                  rotate=1,
#                  offset=False,
#                  ratio=0.5,
#                  mode=1,
#                  prob=0.7,
#                  upper_iter=360000):
#         """
#         GridMask Data Augmentation, see https://arxiv.org/abs/2001.04086
#         Args:
#             use_h (bool): whether to mask vertically
#             use_w (boo;): whether to mask horizontally
#             rotate (float): angle for the mask to rotate
#             offset (float): mask offset
#             ratio (float): mask ratio
#             mode (int): gridmask mode
#             prob (float): max probability to carry out gridmask
#             upper_iter (int): suggested to be equal to global max_iter
#         """
#         super(GridMask, self).__init__()
#         self.use_h = use_h
#         self.use_w = use_w
#         self.rotate = rotate
#         self.offset = offset
#         self.ratio = ratio
#         self.mode = mode
#         self.prob = prob
#         self.upper_iter = upper_iter
#
#         from .gridmask_utils import Gridmask
#         self.gridmask_op = Gridmask(
#             use_h,
#             use_w,
#             rotate=rotate,
#             offset=offset,
#             ratio=ratio,
#             mode=mode,
#             prob=prob,
#             upper_iter=upper_iter)
#
#     def apply(self, sample, context=None):
#         sample['image'] = self.gridmask_op(sample['image'], sample['curr_iter'])
#         return sample


# @register_op
class RandomDistort(BaseOperator):
    """Random color distortion.
    Args:
        hue (list): hue settings. in [lower, upper, probability] format.
        saturation (list): saturation settings. in [lower, upper, probability] format.
        contrast (list): contrast settings. in [lower, upper, probability] format.
        brightness (list): brightness settings. in [lower, upper, probability] format.
        random_apply (bool): whether to apply in random (yolo) or fixed (SSD)
            order.
        count (int): the number of doing distrot
        random_channel (bool): whether to swap channels randomly
    """

    def __init__(self,
                 hue=[-18, 18, 0.5],
                 saturation=[0.5, 1.5, 0.5],
                 contrast=[0.5, 1.5, 0.5],
                 brightness=[0.5, 1.5, 0.5],
                 random_apply=True,
                 count=4,
                 random_channel=False):
        super(RandomDistort, self).__init__()
        self.hue = hue
        self.saturation = saturation
        self.contrast = contrast
        self.brightness = brightness
        self.random_apply = random_apply
        self.count = count
        self.random_channel = random_channel

    def apply_hue(self, img):
        """

        :param img:
        :return:
        """
        low, high, prob = self.hue
        if np.random.uniform(0., 1.) < prob:
            return img

        img = img.astype(np.float32)
        # it works, but result differ from HSV version
        delta = np.random.uniform(low, high)
        u = np.cos(delta * np.pi)
        w = np.sin(delta * np.pi)
        bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
        tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321],
                         [0.211, -0.523, 0.311]])
        ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647],
                          [1.0, -1.107, 1.705]])
        t = np.dot(np.dot(ityiq, bt), tyiq).T
        img = np.dot(img, t)
        return img

    def apply_saturation(self, img):
        """

        :param img:
        :return:
        """
        low, high, prob = self.saturation
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        # it works, but result differ from HSV version
        gray = img * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)
        gray = gray.sum(axis=2, keepdims=True)
        gray *= (1.0 - delta)
        img *= delta
        img += gray
        return img

    def apply_contrast(self, img):
        """

        :param img:
        :return:
        """
        low, high, prob = self.contrast
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        img *= delta
        return img

    def apply_brightness(self, img):
        """
        :param img:
        :return:
        """
        
        low, high, prob = self.brightness
        if np.random.uniform(0., 1.) < prob:
            return img
        delta = np.random.uniform(low, high)
        img = img.astype(np.float32)
        img += delta
        return img

    def apply(self, sample, context=None):
        img = sample['image']
        if self.random_apply:
            functions = [
                self.apply_brightness, self.apply_contrast,
                # self.apply_saturation,
                # self.apply_hue
            ]
            distortions = np.random.permutation(functions)[:self.count]
            for func in distortions:
                img = func(img)
            sample['image'] = img
            return sample

        img = self.apply_brightness(img)
        mode = np.random.randint(0, 2)

        if mode:
            img = self.apply_contrast(img)

        # img = self.apply_saturation(img)
        # img = self.apply_hue(img)

        if not mode:
            img = self.apply_contrast(img)

        # if self.random_channel:
        #     if np.random.randint(0, 2):
        #         img = img[..., np.random.permutation(3)]
        sample['image'] = img
        return sample


# @register_op
# class AutoAugment(BaseOperator):
#     def __init__(self, autoaug_type="v1"):
#         """
#         Args:
#             autoaug_type (str): autoaug type, support v0, v1, v2, v3, test
#         """
#         super(AutoAugment, self).__init__()
#         self.autoaug_type = autoaug_type
#
#     def apply(self, sample, context=None):
#         """
#         Learning Data Augmentation Strategies for Object Detection, see https://arxiv.org/abs/1906.11172
#         """
#         im = sample['image']
#         gt_bbox = sample['gt_bbox']
#         if not isinstance(im, np.ndarray):
#             raise TypeError("{}: image is not a numpy array.".format(self))
#         if len(im.shape) != 3:
#             raise ImageError("{}: image is not 3-dimensional.".format(self))
#         if len(gt_bbox) == 0:
#             return sample
#
#         height, width, _ = im.shape
#         norm_gt_bbox = np.ones_like(gt_bbox, dtype=np.float32)
#         norm_gt_bbox[:, 0] = gt_bbox[:, 1] / float(height)
#         norm_gt_bbox[:, 1] = gt_bbox[:, 0] / float(width)
#         norm_gt_bbox[:, 2] = gt_bbox[:, 3] / float(height)
#         norm_gt_bbox[:, 3] = gt_bbox[:, 2] / float(width)
#
#         from .autoaugment_utils import distort_image_with_autoaugment
#         im, norm_gt_bbox = distort_image_with_autoaugment(im, norm_gt_bbox,
#                                                           self.autoaug_type)
#
#         gt_bbox[:, 0] = norm_gt_bbox[:, 1] * float(width)
#         gt_bbox[:, 1] = norm_gt_bbox[:, 0] * float(height)
#         gt_bbox[:, 2] = norm_gt_bbox[:, 3] * float(width)
#         gt_bbox[:, 3] = norm_gt_bbox[:, 2] * float(height)
#
#         sample['image'] = im
#         sample['gt_bbox'] = gt_bbox
#         return sample


# @register_op
class RandomFlip(BaseOperator):
    """
    Random flip.
    """

    def __init__(self, prob=0.5, use_cloud=False, imgKouZhaos=[]):
        """
        Args:
            prob (float): the probability of flipping image
        """
        super(RandomFlip, self).__init__()
        self.prob = prob
        self.use_cloud = use_cloud
        self.imgKouZhaos = imgKouZhaos
        if not (isinstance(self.prob, float)):
            raise TypeError("{}: input type is invalid.".format(self))

    def apply_segm(self, segms, height, width):
        """

        :param segms:
        :param height:
        :param width:
        :return:
        """

        def _flip_poly(poly, width):
            flipped_poly = np.array(poly)
            flipped_poly[0::2] = width - np.array(poly[0::2])
            return flipped_poly.tolist()

        def _flip_rle(rle, height, width):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            mask = mask[:, ::-1]
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        flipped_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                flipped_segms.append([_flip_poly(poly, width) for poly in segm])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                flipped_segms.append(_flip_rle(segm, height, width))
        return flipped_segms

    def apply_keypoint(self, gt_keypoint, width):
        """

        :param gt_keypoint:
        :param width:
        :return:
        """
        for i in range(gt_keypoint.shape[1]):
            if i % 2 == 0:
                old_x = gt_keypoint[:, i].copy()
                gt_keypoint[:, i] = width - old_x
        return gt_keypoint

    def apply_image(self, image):
        """

        :param image:
        :return:
        """
        im = cv2.flip(image, 1)
        return im

    def apply_bbox(self, bbox, width):
        """

        :param bbox:
        :param width:
        :return:
        """
        oldx1 = bbox[:, 0].copy()
        oldx2 = bbox[:, 2].copy()
        bbox[:, 0] = width - oldx2
        bbox[:, 2] = width - oldx1
        return bbox

    def apply_rbox(self, bbox, width):
        """

        :param bbox:
        :param width:
        :return:
        """
        oldx1 = bbox[:, 0].copy()
        oldx2 = bbox[:, 2].copy()
        oldx3 = bbox[:, 4].copy()
        oldx4 = bbox[:, 6].copy()
        bbox[:, 0] = width - oldx1
        bbox[:, 2] = width - oldx2
        bbox[:, 4] = width - oldx3
        bbox[:, 6] = width - oldx4
        bbox = [bbox_utils.get_best_begin_point_single(e) for e in bbox]
        return bbox

    def apply(self, sample, context=None):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """

        def AddHaze1(img):
            # img_f = img           # 保存文件用这行
            img_f = img / 255.0  # 对原博主的代码进行了更改，这是显示时的样子
            (row, col) = img.shape

            A = 0.5  # 亮度
            beta = 0.08  # 雾的浓度
            beta = 0.04  # 雾的浓度
            size = math.sqrt(max(row, col))  # 雾化尺寸
            center = (row // 2, col // 2)  # 雾化中心
            for j in range(row):
                for l in range(col):
                    d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
                    td = math.exp(-beta * d)
                    img_f[j][l] = img_f[j][l] * td + A * (1 - td)
            return img_f
        # img = sample['image']
        # img = np.array(img)
        # img[img > 255] = 255
        # img[img < 0] = 0
        # img = np.array(img).astype(np.uint8)
        # imgrgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # box = sample['gt_bbox']
        # for i in range(box.shape[0]):
        #     boxt = box[i]
        #     l, t, r, b = map(int, boxt)
        #     cv2.rectangle(imgrgb, (l, t), (r, b), (0, 0, 255), 10, 8, 0)
        # cv2.imwrite("/home/baidu/Desktop/model_img_tmp2/flip_" + str(int(1000 * (time.time()))) + ".jpg", imgrgb)
        
        #对20%的数据做戴口罩操作，循环方框，每个方框有1/2的概率戴口罩
        if random.uniform(0, 1) < 0.4:
            boxtmp = sample['gt_bbox']
            img1 = sample['image']
            imgh,imgw = img1.shape[:2]
            for i in range(boxtmp.shape[0]):
                if random.uniform(0, 1) < 0.5:
                    continue
                l,t,r,b = boxtmp[i]
                l,t,r,b = int(l),int(t),int(r),int(b)
                if r>= imgw or b >= imgh:
                    continue
                img_kouzhao_id = random.randint(0, len(self.imgKouZhaos)-1)
                img_kouzhao = cv.imread(self.imgKouZhaos[img_kouzhao_id], 0)
                w = r-l
                fratio1 = random.uniform(0.4, 0.6)
                fratio1 = random.uniform(0.4, 0.8)
                h = int(b*fratio1-t*fratio1)
                if w == 0 or h == 0:
                    continue
                img_kouzhao1 = cv.resize(img_kouzhao, (w,h))
                try:
                # print ("kouzhao:", w, h, img_kouzhao1.shape)
                    img1[b-h:b,r-w:r] = img_kouzhao1
                except:
                    ""
            sample['image'] = img1
        
        #对10%的数据进行雾化操作
        if random.uniform(0, 1) < 0.15:
            img1 = sample['image']
            img2 = AddHaze1(img1)
            img3 = np.array(img2*255, dtype=np.uint8)
            sample['image'] = img3        

        if np.random.uniform(0, 1) < self.prob:
            return sample

        if 'gt_bbox' not in sample:
            return sample
        if sample['gt_bbox'] is None or len(sample['gt_bbox']) == 0:
            return sample

        im = sample['image']
        height, width = im.shape[:2]
        im = self.apply_image(im)
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], width)
        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(sample['gt_poly'], height,
                                                width)
        if 'gt_keypoint' in sample and len(sample['gt_keypoint']) > 0:
            sample['gt_keypoint'] = self.apply_keypoint(
                sample['gt_keypoint'], width)

        if 'semantic' in sample and sample['semantic']:
            sample['semantic'] = sample['semantic'][:, ::-1]

        if 'gt_segm' in sample and sample['gt_segm'].any():
            sample['gt_segm'] = sample['gt_segm'][:, :, ::-1]

        if 'gt_rbox2poly' in sample and sample['gt_rbox2poly'].any():
            sample['gt_rbox2poly'] = self.apply_rbox(sample['gt_rbox2poly'],
                                                     width)
        if random.randint(0, 500) == 1:
            img = copy.deepcopy(im)
            img = np.array(img)
            img[img > 255] = 255
            img[img < 0] = 0
            img = np.array(img).astype(np.uint8)
            imgrgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if self.use_cloud:
                if sample['only_negimg']:
                    cv2.imwrite("/root/paddlejob/data_train/train_mid_img/only_negimg_" + str(
                        int(1000 * (time.time()))) + ".jpg", imgrgb)
                else:
                    box = sample['gt_bbox']
                    for i in range(box.shape[0]):
                        boxt = box[i]
                        l, t, r, b = map(int, boxt)
                        cv2.rectangle(imgrgb, (l, t), (r, b), (0, 0, 255), 10, 8, 0)
                    cv2.imwrite(
                        "/root/paddlejob/data_train/train_mid_img/flip_" + str(int(1000 * (time.time()))) + ".jpg",
                        imgrgb)
            else:
                if sample['only_negimg']:
                    cv2.imwrite("/home/yubin/Desktop/6faces_2/only_negimg_" + str(int(1000 * (time.time()))) + ".jpg",
                                imgrgb)
                else:
                    box = sample['gt_bbox']
                    for i in range(box.shape[0]):
                        boxt = box[i]
                        l, t, r, b = map(int, boxt)
                        cv2.rectangle(imgrgb, (l, t), (r, b), (0, 0, 255), 10, 8, 0)
                    cv2.imwrite("/home/yubin/Desktop/6faces_2/flip_" + str(int(1000 * (time.time()))) + ".jpg", imgrgb)
        # cv2.imwrite("/home/baidu/Desktop/6hands/im_" + str(int(1000 * (time.time()))) + ".jpg", im[:,:,::-1])
        sample['flipped'] = True
        sample['image'] = im

        return sample


# 上下反转
class RandomFlip_Ver(BaseOperator):
    """
    Random flip.
    """

    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): the probability of flipping image
        """
        super(RandomFlip_Ver, self).__init__()
        self.prob = prob
        if not (isinstance(self.prob, float)):
            raise TypeError("{}: input type is invalid.".format(self))

    def apply_image(self, image):
        """

        :param image:
        :return:
        """
        return image[::-1, :, :]

    def apply_bbox(self, bbox, height):
        """

        :param bbox:
        :param width:
        :return:
        """
        oldy1 = bbox[:, 1].copy()
        oldy2 = bbox[:, 3].copy()
        bbox[:, 1] = height - oldy2
        bbox[:, 3] = height - oldy1
        return bbox

    def apply(self, sample, context=None):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """
        if np.random.uniform(0, 1) < self.prob:
            return sample

        if 'gt_bbox' not in sample:
            return sample
        if sample['gt_bbox'] is None or len(sample['gt_bbox']) == 0:
            return sample

        im = sample['image']
        height, width = im.shape[:2]
        im = self.apply_image(im)
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], height)

        sample['flipped'] = True
        sample['image'] = im

        return sample


# @register_op
class Resize(BaseOperator):
    """
    Resize image to target size. if keep_ratio is True,
    resize the image's long side to the maximum of target_size
    if keep_ratio is False, resize the image to target size(h, w)
    """

    def __init__(self, target_size, keep_ratio, interp=cv2.INTER_LINEAR):
        """
        Resize image to target size. if keep_ratio is True, 
        resize the image's long side to the maximum of target_size
        if keep_ratio is False, resize the image to target size(h, w)
        Args:
            target_size (int|list): image target size
            keep_ratio (bool): whether keep_ratio or not, default true
            interp (int): the interpolation method
        """
        super(Resize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interp = interp
        if not isinstance(target_size, (Integral, Sequence)):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List or Tuple, now is {}".
                    format(type(target_size)))
        if isinstance(target_size, Integral):
            target_size = [target_size, target_size]
        self.target_size = target_size

    def apply_image(self, image, scale):
        """

        :param image:
        :param scale:
        :return:
        """
        im_scale_x, im_scale_y = scale

        return cv2.resize(
            image,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp)

    def apply_bbox(self, bbox, scale, size):
        """

        :param bbox:
        :param scale:
        :param size:
        :return:
        """
        im_scale_x, im_scale_y = scale
        resize_w, resize_h = size
        bbox[:, 0::2] *= im_scale_x
        bbox[:, 1::2] *= im_scale_y
        bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, resize_w)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, resize_h)
        return bbox

    def apply_segm(self, segms, im_size, scale):
        """

        :param segms:
        :param im_size:
        :param scale:
        :return:
        """

        def _resize_poly(poly, im_scale_x, im_scale_y):
            resized_poly = np.array(poly)
            resized_poly[0::2] *= im_scale_x
            resized_poly[1::2] *= im_scale_y
            return resized_poly.tolist()

        def _resize_rle(rle, im_h, im_w, im_scale_x, im_scale_y):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, im_h, im_w)

            mask = mask_util.decode(rle)
            mask = cv2.resize(
                mask,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        im_h, im_w = im_size
        im_scale_x, im_scale_y = scale
        resized_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                resized_segms.append([
                    _resize_poly(poly, im_scale_x, im_scale_y) for poly in segm
                ])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                resized_segms.append(
                    _resize_rle(segm, im_h, im_w, im_scale_x, im_scale_y))

        return resized_segms

    def apply(self, sample, context=None):
        """ Resize the image numpy.
        """
        im = sample['image']
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        # if len(im.shape) != 3:
        #     raise ImageError('{}: image is not 3-dimensional.'.format(self))

        # apply image
        im_shape = im.shape
        if self.keep_ratio:

            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])

            target_size_min = np.min(self.target_size)
            target_size_max = np.max(self.target_size)

            im_scale = min(target_size_min / im_size_min,
                           target_size_max / im_size_max)

            resize_h = im_scale * float(im_shape[0])
            resize_w = im_scale * float(im_shape[1])

            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            resize_h, resize_w = self.target_size
            im_scale_y = resize_h / im_shape[0]
            im_scale_x = resize_w / im_shape[1]

        im = self.apply_image(sample['image'], [im_scale_x, im_scale_y])
        sample['image'] = im
        sample['im_shape'] = np.asarray([resize_h, resize_w], dtype=np.float32)
        if 'scale_factor' in sample:
            scale_factor = sample['scale_factor']
            sample['scale_factor'] = np.asarray(
                [scale_factor[0] * im_scale_y, scale_factor[1] * im_scale_x],
                dtype=np.float32)
        else:
            sample['scale_factor'] = np.asarray(
                [im_scale_y, im_scale_x], dtype=np.float32)

        # apply bbox
        if 'gt_bbox' in sample:
            if sample['gt_bbox'] is None or len(sample['gt_bbox']) == 0:
                ""
            else:
                sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'],
                                                    [im_scale_x, im_scale_y],
                                                    [resize_w, resize_h])

        # apply rbox
        if 'gt_rbox2poly' in sample:
            if np.array(sample['gt_rbox2poly']).shape[1] != 8:
                logger.warn(
                    "gt_rbox2poly's length shoule be 8, but actually is {}".
                        format(len(sample['gt_rbox2poly'])))
            sample['gt_rbox2poly'] = self.apply_bbox(sample['gt_rbox2poly'],
                                                     [im_scale_x, im_scale_y],
                                                     [resize_w, resize_h])

        # apply polygon
        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(sample['gt_poly'], im_shape[:2],
                                                [im_scale_x, im_scale_y])

        # apply semantic
        if 'semantic' in sample and sample['semantic']:
            semantic = sample['semantic']
            semantic = cv2.resize(
                semantic.astype('float32'),
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
            semantic = np.asarray(semantic).astype('int32')
            semantic = np.expand_dims(semantic, 0)
            sample['semantic'] = semantic

        # apply gt_segm
        if 'gt_segm' in sample and len(sample['gt_segm']) > 0:
            masks = [
                cv2.resize(
                    gt_segm,
                    None,
                    None,
                    fx=im_scale_x,
                    fy=im_scale_y,
                    interpolation=cv2.INTER_NEAREST)
                for gt_segm in sample['gt_segm']
            ]
            sample['gt_segm'] = np.asarray(masks).astype(np.uint8)

        return sample


# @register_op
class MultiscaleTestResize(BaseOperator):
    """
    Rescale image to the each size in target size, and capped at max_size.
    """

    def __init__(self,
                 origin_target_size=[800, 1333],
                 target_size=[],
                 interp=cv2.INTER_LINEAR,
                 use_flip=True):
        """
        Rescale image to the each size in target size, and capped at max_size.
        Args:
            origin_target_size (list): origin target size of image
            target_size (list): A list of target sizes of image.
            interp (int): the interpolation method.
            use_flip (bool): whether use flip augmentation.
        """
        super(MultiscaleTestResize, self).__init__()
        self.interp = interp
        self.use_flip = use_flip

        if not isinstance(target_size, Sequence):
            raise TypeError(
                "Type of target_size is invalid. Must be List or Tuple, now is {}".
                    format(type(target_size)))
        self.target_size = target_size

        if not isinstance(origin_target_size, Sequence):
            raise TypeError(
                "Type of origin_target_size is invalid. Must be List or Tuple, now is {}".
                    format(type(origin_target_size)))

        self.origin_target_size = origin_target_size

    def apply(self, sample, context=None):
        """ Resize the image numpy for multi-scale test.
        """
        samples = []
        resizer = Resize(
            self.origin_target_size, keep_ratio=True, interp=self.interp)
        samples.append(resizer(sample.copy(), context))
        if self.use_flip:
            flipper = RandomFlip(1.1)
            samples.append(flipper(sample.copy(), context=context))

        for size in self.target_size:
            resizer = Resize(size, keep_ratio=True, interp=self.interp)
            samples.append(resizer(sample.copy(), context))

        return samples


# @register_op
class RandomResize(BaseOperator):
    """
    Resize image to target size randomly. random target_size and interpolation method
    """

    def __init__(self,
                 target_size,
                 keep_ratio=True,
                 interp=cv2.INTER_LINEAR,
                 random_size=True,
                 random_interp=False):
        """
        Resize image to target size randomly. random target_size and interpolation method
        Args:
            target_size (int, list, tuple): image target size, if random size is True, must be list or tuple
            keep_ratio (bool): whether keep_raio or not, default true
            interp (int): the interpolation method
            random_size (bool): whether random select target size of image
            random_interp (bool): whether random select interpolation method
        """
        super(RandomResize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interp = interp
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]
        assert isinstance(target_size, (
            Integral, Sequence)), "target_size must be Integer, List or Tuple"
        if random_size and not isinstance(target_size, Sequence):
            raise TypeError(
                "Type of target_size is invalid when random_size is True. Must be List or Tuple, now is {}".
                    format(type(target_size)))
        self.target_size = target_size
        self.random_size = random_size
        self.random_interp = random_interp

    def apply(self, sample, context=None):
        """ Resize the image numpy.
        """
        if self.random_size:
            target_size = random.choice(self.target_size)
        else:
            target_size = self.target_size

        if self.random_interp:
            interp = random.choice(self.interps)
        else:
            interp = self.interp

        resizer = Resize(target_size, self.keep_ratio, interp)
        return resizer(sample, context=context)


# @register_op
class RandomExpand(BaseOperator):
    """Random expand the canvas.
    Args:
        ratio (float): maximum expansion ratio.
        prob (float): probability to expand.
        fill_value (list): color value used to fill the canvas. in RGB order.
    """

    def __init__(self, ratio=4., prob=0.5, fill_value=(127.5)):
        super(RandomExpand, self).__init__()
        assert ratio > 1.01, "expand ratio must be larger than 1.01"
        self.ratio = ratio
        self.prob = prob
        assert isinstance(fill_value, (Number, Sequence)), \
            "fill value must be either float or sequence"
        if isinstance(fill_value, Number):
            fill_value = (fill_value,) * 3
        if not isinstance(fill_value, tuple):
            fill_value = tuple(fill_value)
        self.fill_value = fill_value

    def apply(self, sample, context=None):
        if np.random.uniform(0., 1.) < self.prob:
            return sample

        if sample['crop_MaxHand']:
            return sample

        im = sample['image']
        height, width = im.shape[:2]
        if random.uniform(0,1)<0.6:
            ratio = np.random.uniform(1., self.ratio)
            h = int(height * ratio)
            w = int(width * ratio)
        else:
            ratio1 = np.random.uniform(1., self.ratio)
            h = int(height * ratio1)
            ratio2 = np.random.uniform(1., self.ratio)
            w = int(width * ratio2)
        if not h > height or not w > width:
            return sample
        y = np.random.randint(0, h - height)
        x = np.random.randint(0, w - width)
        offsets, size = [x, y], [h, w]

        pad = Pad(size,
                  pad_mode=-1,
                  offsets=offsets,
                  fill_value=self.fill_value)

        return pad(sample, context=context)


# @register_op
class CropWithSampling(BaseOperator):
    """
    Crop image with sampling.
    """

    def __init__(self, batch_sampler, satisfy_all=False, avoid_no_bbox=True):
        """
        Args:
            batch_sampler (list): Multiple sets of different
                                  parameters for cropping.
            satisfy_all (bool): whether all boxes must satisfy.
            e.g.[[1, 1, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
                 [1, 50, 0.3, 1.0, 0.5, 2.0, 0.1, 1.0],
                 [1, 50, 0.3, 1.0, 0.5, 2.0, 0.3, 1.0],
                 [1, 50, 0.3, 1.0, 0.5, 2.0, 0.5, 1.0],
                 [1, 50, 0.3, 1.0, 0.5, 2.0, 0.7, 1.0],
                 [1, 50, 0.3, 1.0, 0.5, 2.0, 0.9, 1.0],
                 [1, 50, 0.3, 1.0, 0.5, 2.0, 0.0, 1.0]]
           [max sample, max trial, min scale, max scale,
            min aspect ratio, max aspect ratio,
            min overlap, max overlap]
            avoid_no_bbox (bool): whether to to avoid the
                                  situation where the box does not appear.
        """
        super(CropWithSampling, self).__init__()
        self.batch_sampler = batch_sampler
        self.satisfy_all = satisfy_all
        self.avoid_no_bbox = avoid_no_bbox

    def apply(self, sample, context):
        """
        Crop the image and modify bounding box.
        Operators:
            1. Scale the image width and height.
            2. Crop the image according to a radom sample.
            3. Rescale the bounding box.
            4. Determine if the new bbox is satisfied in the new image.
        Returns:
            sample: the image, bounding box are replaced.
        """
        assert 'image' in sample, "image data not found"
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']
        im_height, im_width = im.shape[:2]
        gt_score = None
        if 'gt_score' in sample:
            gt_score = sample['gt_score']
        sampled_bbox = []
        gt_bbox = gt_bbox.tolist()
        for sampler in self.batch_sampler:
            found = 0
            for i in range(sampler[1]):
                if found >= sampler[0]:
                    break
                sample_bbox = generate_sample_bbox(sampler)
                if satisfy_sample_constraint(sampler, sample_bbox, gt_bbox,
                                             self.satisfy_all):
                    sampled_bbox.append(sample_bbox)
                    found = found + 1
        im = np.array(im)
        while sampled_bbox:
            idx = int(np.random.uniform(0, len(sampled_bbox)))
            sample_bbox = sampled_bbox.pop(idx)
            sample_bbox = clip_bbox(sample_bbox)
            crop_bbox, crop_class, crop_score = \
                filter_and_process(sample_bbox, gt_bbox, gt_class, scores=gt_score)
            if self.avoid_no_bbox:
                if len(crop_bbox) < 1:
                    continue
            xmin = int(sample_bbox[0] * im_width)
            xmax = int(sample_bbox[2] * im_width)
            ymin = int(sample_bbox[1] * im_height)
            ymax = int(sample_bbox[3] * im_height)
            im = im[ymin:ymax, xmin:xmax]
            sample['image'] = im
            sample['gt_bbox'] = crop_bbox
            sample['gt_class'] = crop_class
            sample['gt_score'] = crop_score
            return sample
        return sample


# @register_op
class CropWithDataAchorSampling(BaseOperator):
    """
    Crop image with data anchor sampling.
    """

    def __init__(self,
                 batch_sampler,
                 anchor_sampler=None,
                 target_size=None,
                 das_anchor_scales=[16, 32, 64, 128],
                 sampling_prob=0.5,
                 min_size=8.,
                 avoid_no_bbox=True):
        """
        Args:
            anchor_sampler (list): anchor_sampling sets of different
                                  parameters for cropping.
            batch_sampler (list): Multiple sets of different
                                  parameters for cropping.
              e.g.[[1, 10, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.2, 0.0]]
                  [[1, 50, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                   [1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                   [1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                   [1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                   [1, 50, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]]
              [max sample, max trial, min scale, max scale,
               min aspect ratio, max aspect ratio,
               min overlap, max overlap, min coverage, max coverage]
            target_size (int): target image size.
            das_anchor_scales (list[float]): a list of anchor scales in data
                anchor smapling.
            min_size (float): minimum size of sampled bbox.
            avoid_no_bbox (bool): whether to to avoid the
                                  situation where the box does not appear.
        """
        super(CropWithDataAchorSampling, self).__init__()
        self.anchor_sampler = anchor_sampler
        self.batch_sampler = batch_sampler
        self.target_size = target_size
        self.sampling_prob = sampling_prob
        self.min_size = min_size
        self.avoid_no_bbox = avoid_no_bbox
        self.das_anchor_scales = np.array(das_anchor_scales)

    def apply(self, sample, context):
        """
        Crop the image and modify bounding box.
        Operators:
            1. Scale the image width and height.
            2. Crop the image according to a radom sample.
            3. Rescale the bounding box.
            4. Determine if the new bbox is satisfied in the new image.
        Returns:
            sample: the image, bounding box are replaced.
        """
        assert 'image' in sample, "image data not found"
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']
        image_height, image_width = im.shape[:2]
        gt_bbox[:, 0] /= image_width
        gt_bbox[:, 1] /= image_height
        gt_bbox[:, 2] /= image_width
        gt_bbox[:, 3] /= image_height
        gt_score = None
        if 'gt_score' in sample:
            gt_score = sample['gt_score']
        sampled_bbox = []
        gt_bbox = gt_bbox.tolist()

        prob = np.random.uniform(0., 1.)
        if prob > self.sampling_prob:  # anchor sampling
            assert self.anchor_sampler
            for sampler in self.anchor_sampler:
                found = 0
                for i in range(sampler[1]):
                    if found >= sampler[0]:
                        break
                    sample_bbox = data_anchor_sampling(
                        gt_bbox, image_width, image_height,
                        self.das_anchor_scales, self.target_size)
                    if sample_bbox == 0:
                        break
                    if satisfy_sample_constraint_coverage(sampler, sample_bbox,
                                                          gt_bbox):
                        sampled_bbox.append(sample_bbox)
                        found = found + 1
            im = np.array(im)
            while sampled_bbox:
                idx = int(np.random.uniform(0, len(sampled_bbox)))
                sample_bbox = sampled_bbox.pop(idx)

                if 'gt_keypoint' in sample.keys():
                    keypoints = (sample['gt_keypoint'],
                                 sample['keypoint_ignore'])
                    crop_bbox, crop_class, crop_score, gt_keypoints = \
                        filter_and_process(sample_bbox, gt_bbox, gt_class,
                                           scores=gt_score,
                                           keypoints=keypoints)
                else:
                    crop_bbox, crop_class, crop_score = filter_and_process(
                        sample_bbox, gt_bbox, gt_class, scores=gt_score)
                crop_bbox, crop_class, crop_score = bbox_area_sampling(
                    crop_bbox, crop_class, crop_score, self.target_size,
                    self.min_size)

                if self.avoid_no_bbox:
                    if len(crop_bbox) < 1:
                        continue
                im = crop_image_sampling(im, sample_bbox, image_width,
                                         image_height, self.target_size)
                height, width = im.shape[:2]
                crop_bbox[:, 0] *= width
                crop_bbox[:, 1] *= height
                crop_bbox[:, 2] *= width
                crop_bbox[:, 3] *= height
                sample['image'] = im
                sample['gt_bbox'] = crop_bbox
                sample['gt_class'] = crop_class
                if 'gt_score' in sample:
                    sample['gt_score'] = crop_score
                if 'gt_keypoint' in sample.keys():
                    sample['gt_keypoint'] = gt_keypoints[0]
                    sample['keypoint_ignore'] = gt_keypoints[1]
                return sample
            return sample

        else:
            for sampler in self.batch_sampler:
                found = 0
                for i in range(sampler[1]):
                    if found >= sampler[0]:
                        break
                    sample_bbox = generate_sample_bbox_square(
                        sampler, image_width, image_height)
                    if satisfy_sample_constraint_coverage(sampler, sample_bbox,
                                                          gt_bbox):
                        sampled_bbox.append(sample_bbox)
                        found = found + 1
            im = np.array(im)
            while sampled_bbox:
                idx = int(np.random.uniform(0, len(sampled_bbox)))
                sample_bbox = sampled_bbox.pop(idx)
                sample_bbox = clip_bbox(sample_bbox)

                if 'gt_keypoint' in sample.keys():
                    keypoints = (sample['gt_keypoint'],
                                 sample['keypoint_ignore'])
                    crop_bbox, crop_class, crop_score, gt_keypoints = \
                        filter_and_process(sample_bbox, gt_bbox, gt_class,
                                           scores=gt_score,
                                           keypoints=keypoints)
                else:
                    crop_bbox, crop_class, crop_score = filter_and_process(
                        sample_bbox, gt_bbox, gt_class, scores=gt_score)
                # sampling bbox according the bbox area
                crop_bbox, crop_class, crop_score = bbox_area_sampling(
                    crop_bbox, crop_class, crop_score, self.target_size,
                    self.min_size)

                if self.avoid_no_bbox:
                    if len(crop_bbox) < 1:
                        continue
                xmin = int(sample_bbox[0] * image_width)
                xmax = int(sample_bbox[2] * image_width)
                ymin = int(sample_bbox[1] * image_height)
                ymax = int(sample_bbox[3] * image_height)
                im = im[ymin:ymax, xmin:xmax]
                height, width = im.shape[:2]
                crop_bbox[:, 0] *= width
                crop_bbox[:, 1] *= height
                crop_bbox[:, 2] *= width
                crop_bbox[:, 3] *= height
                sample['image'] = im
                sample['gt_bbox'] = crop_bbox
                sample['gt_class'] = crop_class
                if 'gt_score' in sample:
                    sample['gt_score'] = crop_score
                if 'gt_keypoint' in sample.keys():
                    sample['gt_keypoint'] = gt_keypoints[0]
                    sample['keypoint_ignore'] = gt_keypoints[1]
                return sample
            return sample


# @register_op
class RandomCrop(BaseOperator):
    """Random crop image and bboxes.
    Args:
        aspect_ratio (list): aspect ratio of cropped region.
            in [min, max] format.
        thresholds (list): iou thresholds for decide a valid bbox crop.
        scaling (list): ratio between a cropped region and the original image.
             in [min, max] format.
        num_attempts (int): number of tries before giving up.
        allow_no_crop (bool): allow return without actually cropping them.
        cover_all_box (bool): ensure all bboxes are covered in the final crop.
        is_mask_crop(bool): whether crop the segmentation.
    """

    def __init__(self,
                 aspect_ratio=[.5, 2.],
                 # thresholds=[.0, .1, .3, .5, .7, .9],
                 thresholds=[.0, .1, .3, .5],
                 scaling=[.3, 1.],
                 num_attempts=50,
                 allow_no_crop=True,
                 cover_all_box=False,
                 is_mask_crop=False):
        super(RandomCrop, self).__init__()
        self.aspect_ratio = aspect_ratio
        self.thresholds = thresholds
        self.scaling = scaling
        self.num_attempts = num_attempts
        self.allow_no_crop = allow_no_crop
        self.cover_all_box = cover_all_box
        self.is_mask_crop = is_mask_crop

    def crop_segms(self, segms, valid_ids, crop, height, width):
        """

        :param segms:
        :param valid_ids:
        :param crop:
        :param height:
        :param width:
        :return:
        """

        def _crop_poly(segm, crop):
            xmin, ymin, xmax, ymax = crop
            crop_coord = [xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin]
            crop_p = np.array(crop_coord).reshape(4, 2)
            crop_p = Polygon(crop_p)

            crop_segm = list()
            for poly in segm:
                poly = np.array(poly).reshape(len(poly) // 2, 2)
                polygon = Polygon(poly)
                if not polygon.is_valid:
                    exterior = polygon.exterior
                    multi_lines = exterior.intersection(exterior)
                    polygons = shapely.ops.polygonize(multi_lines)
                    polygon = MultiPolygon(polygons)
                multi_polygon = list()
                if isinstance(polygon, MultiPolygon):
                    multi_polygon = copy.deepcopy(polygon)
                else:
                    multi_polygon.append(copy.deepcopy(polygon))
                for per_polygon in multi_polygon:
                    inter = per_polygon.intersection(crop_p)
                    if not inter:
                        continue
                    if isinstance(inter, (MultiPolygon, GeometryCollection)):
                        for part in inter:
                            if not isinstance(part, Polygon):
                                continue
                            part = np.squeeze(
                                np.array(part.exterior.coords[:-1]).reshape(1,
                                                                            -1))
                            part[0::2] -= xmin
                            part[1::2] -= ymin
                            crop_segm.append(part.tolist())
                    elif isinstance(inter, Polygon):
                        crop_poly = np.squeeze(
                            np.array(inter.exterior.coords[:-1]).reshape(1, -1))
                        crop_poly[0::2] -= xmin
                        crop_poly[1::2] -= ymin
                        crop_segm.append(crop_poly.tolist())
                    else:
                        continue
            return crop_segm

        def _crop_rle(rle, crop, height, width):
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            mask = mask[crop[1]:crop[3], crop[0]:crop[2]]
            rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
            return rle

        crop_segms = []
        for id in valid_ids:
            segm = segms[id]
            if is_poly(segm):
                import copy
                import logging
                import shapely.ops
                from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
                logging.getLogger("shapely").setLevel(logging.WARNING)
                # Polygon format
                crop_segms.append(_crop_poly(segm, crop))
            else:
                # RLE format
                import pycocotools.mask as mask_util
                crop_segms.append(_crop_rle(segm, crop, height, width))
        return crop_segms

    def apply(self, sample, context=None):
        if 'gt_bbox' not in sample:
            return sample
        if sample['gt_bbox'] is None or len(sample['gt_bbox']) == 0:
            return sample

        if 'gt_bbox' in sample and len(sample['gt_bbox']) == 0:
            return sample

        if sample['crop_MaxHand']:
            result1 = copy.deepcopy(sample)
            img_tmp1 = sample['image']
            img_h, img_w, _ = img_tmp1.shape
            bbox_tmp1 = sample['gt_bbox']
            l, t, r, b = list(bbox_tmp1[0])
            box_w = r - l
            box_h = b - t
            centx = (l + r) / 2.
            centy = (t + b) / 2.
            fscale = random.uniform(1.2, 3.5)
            box_w_new = fscale * box_w
            box_h_new = fscale * box_h
            l_new = max(0, centx - box_w_new / 2.)
            r_new = min(img_w, centx + box_w_new / 2.)
            t_new = max(0, centy - box_h_new / 2.)
            b_new = min(img_h, centy + box_h_new / 2.)
            l_new, t_new, r_new, b_new = map(int, [l_new, t_new, r_new, b_new])
            img_tmp2 = img_tmp1[t_new:b_new, l_new:r_new, :]
            l_dst = l - l_new
            t_dst = t - t_new
            r_dst = r - l_new
            b_dst = b - t_new
            bbox_tmp2 = np.array([[l_dst, t_dst, r_dst, b_dst]]).astype(np.float32)
            result1['image'] = img_tmp2
            result1['gt_bbox'] = bbox_tmp2

            # imgdraw = copy.deepcopy(img_tmp2)
            # l_draw, t_draw, r_draw, b_draw = map(int, [l_dst, t_dst, r_dst, b_dst])
            # cv2.rectangle(imgdraw, (l_draw, t_draw), (r_draw, b_draw), (0,0,255), 3, 8, 0)
            # cv2.imwrite("/media/baidu/ssd2/ppyolo/6w_data/small_test/less_handNum6/newImage/" + str(int(time.time())) + ".jpg", imgdraw[:, :, ::-1] )

            return result1

        h, w = sample['image'].shape[:2]
        gt_bbox = sample['gt_bbox']

        # NOTE Original method attempts to generate one candidate for each
        # threshold then randomly sample one from the resulting list.
        # Here a short circuit approach is taken, i.e., randomly choose a
        # threshold and attempt to find a valid crop, and simply return the
        # first one found.
        # The probability is not exactly the same, kinda resembling the
        # "Monty Hall" problem. Actually carrying out the attempts will affect
        # observability (just like opening doors in the "Monty Hall" game).
        thresholds = list(self.thresholds)
        if self.allow_no_crop:
            thresholds.append('no_crop')
            thresholds.append('no_crop')
        np.random.shuffle(thresholds)

        for thresh in thresholds:
            if thresh == 'no_crop':
                return sample

            found = False
            for i in range(self.num_attempts):
                scale = np.random.uniform(*self.scaling)
                if self.aspect_ratio is not None:
                    min_ar, max_ar = self.aspect_ratio
                    aspect_ratio = np.random.uniform(
                        max(min_ar, scale ** 2), min(max_ar, scale ** -2))
                    h_scale = scale / np.sqrt(aspect_ratio)
                    w_scale = scale * np.sqrt(aspect_ratio)
                else:
                    h_scale = np.random.uniform(*self.scaling)
                    w_scale = np.random.uniform(*self.scaling)
                crop_h = h * h_scale
                crop_w = w * w_scale
                if self.aspect_ratio is None:
                    if crop_h / crop_w < 0.5 or crop_h / crop_w > 2.0:
                        continue

                crop_h = int(crop_h)
                crop_w = int(crop_w)
                crop_y = np.random.randint(0, h - crop_h)
                crop_x = np.random.randint(0, w - crop_w)
                crop_box = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]
                iou = self._iou_matrix(
                    gt_bbox, np.array(
                        [crop_box], dtype=np.float32))
                if iou.max() < thresh:
                    continue

                if self.cover_all_box and iou.min() < thresh:
                    continue

                cropped_box, valid_ids = self._crop_box_with_center_constraint(
                    gt_bbox, np.array(
                        crop_box, dtype=np.float32))
                if valid_ids.size > 0:
                    found = True
                    break

            if found:
                if self.is_mask_crop and 'gt_poly' in sample and len(sample[
                                                                         'gt_poly']) > 0:
                    crop_polys = self.crop_segms(
                        sample['gt_poly'],
                        valid_ids,
                        np.array(
                            crop_box, dtype=np.int64),
                        h,
                        w)
                    if [] in crop_polys:
                        delete_id = list()
                        valid_polys = list()
                        for id, crop_poly in enumerate(crop_polys):
                            if crop_poly == []:
                                delete_id.append(id)
                            else:
                                valid_polys.append(crop_poly)
                        valid_ids = np.delete(valid_ids, delete_id)
                        if len(valid_polys) == 0:
                            return sample
                        sample['gt_poly'] = valid_polys
                    else:
                        sample['gt_poly'] = crop_polys

                if 'gt_segm' in sample:
                    sample['gt_segm'] = self._crop_segm(sample['gt_segm'],
                                                        crop_box)
                    sample['gt_segm'] = np.take(
                        sample['gt_segm'], valid_ids, axis=0)

                sample['image'] = self._crop_image(sample['image'], crop_box)
                sample['gt_bbox'] = np.take(cropped_box, valid_ids, axis=0)
                sample['gt_class'] = np.take(
                    sample['gt_class'], valid_ids, axis=0)
                if 'gt_score' in sample:
                    sample['gt_score'] = np.take(
                        sample['gt_score'], valid_ids, axis=0)

                if 'is_crowd' in sample:
                    sample['is_crowd'] = np.take(
                        sample['is_crowd'], valid_ids, axis=0)
                return sample

        return sample

    def _iou_matrix(self, a, b):
        tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

        area_i = np.prod(br_i - tl_i, axis=2) * (tl_i < br_i).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
        area_o = (area_a[:, np.newaxis] + area_b - area_i)
        return area_i / (area_o + 1e-10)

    def _crop_box_with_center_constraint(self, box, crop):
        cropped_box = box.copy()

        cropped_box[:, :2] = np.maximum(box[:, :2], crop[:2])
        cropped_box[:, 2:] = np.minimum(box[:, 2:], crop[2:])
        cropped_box[:, :2] -= crop[:2]
        cropped_box[:, 2:] -= crop[:2]

        centers = (box[:, :2] + box[:, 2:]) / 2
        valid = np.logical_and(crop[:2] <= centers,
                               centers < crop[2:]).all(axis=1)
        valid = np.logical_and(
            valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))

        return cropped_box, np.where(valid)[0]

    def _crop_image(self, img, crop):
        x1, y1, x2, y2 = crop
        return img[y1:y2, x1:x2, :]

    def _crop_segm(self, segm, crop):
        x1, y1, x2, y2 = crop
        return segm[:, y1:y2, x1:x2]


# @register_op
class RandomScaledCrop(BaseOperator):
    """Resize image and bbox based on long side (with optional random scaling),
       then crop or pad image to target size.
    Args:
        target_dim (int): target size.
        scale_range (list): random scale range.
        interp (int): interpolation method, default to `cv2.INTER_LINEAR`.
    """

    def __init__(self,
                 target_dim=512,
                 scale_range=[.1, 2.],
                 interp=cv2.INTER_LINEAR):
        super(RandomScaledCrop, self).__init__()
        self.target_dim = target_dim
        self.scale_range = scale_range
        self.interp = interp

    def apply(self, sample, context=None):
        img = sample['image']
        h, w = img.shape[:2]
        random_scale = np.random.uniform(*self.scale_range)
        dim = self.target_dim
        random_dim = int(dim * random_scale)
        dim_max = max(h, w)
        scale = random_dim / dim_max
        resize_w = w * scale
        resize_h = h * scale
        offset_x = int(max(0, np.random.uniform(0., resize_w - dim)))
        offset_y = int(max(0, np.random.uniform(0., resize_h - dim)))

        img = cv2.resize(img, (resize_w, resize_h), interpolation=self.interp)
        img = np.array(img)
        canvas = np.zeros((dim, dim, 3), dtype=img.dtype)
        canvas[:min(dim, resize_h), :min(dim, resize_w), :] = img[
                                                              offset_y:offset_y + dim, offset_x:offset_x + dim, :]
        sample['image'] = canvas
        sample['im_shape'] = np.asarray([resize_h, resize_w], dtype=np.float32)
        scale_factor = sample['sacle_factor']
        sample['scale_factor'] = np.asarray(
            [scale_factor[0] * scale, scale_factor[1] * scale],
            dtype=np.float32)

        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            scale_array = np.array([scale, scale] * 2, dtype=np.float32)
            shift_array = np.array([offset_x, offset_y] * 2, dtype=np.float32)
            boxes = sample['gt_bbox'] * scale_array - shift_array
            boxes = np.clip(boxes, 0, dim - 1)
            # filter boxes with no area
            area = np.prod(boxes[..., 2:] - boxes[..., :2], axis=1)
            valid = (area > 1.).nonzero()[0]
            sample['gt_bbox'] = boxes[valid]
            sample['gt_class'] = sample['gt_class'][valid]

        return sample


# @register_op
class Cutmix(BaseOperator):
    """
    CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features,
    see https://arxiv.org/abs/1905.04899
    Cutmix image and gt_bbbox/gt_score
    """

    def __init__(self, alpha=1.5, beta=1.5):
        """ 
        CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features,
        see https://arxiv.org/abs/1905.04899
        Cutmix image and gt_bbbox/gt_score
        Args:
             alpha (float): alpha parameter of beta distribute
             beta (float): beta parameter of beta distribute
        """
        super(Cutmix, self).__init__()
        self.alpha = alpha
        self.beta = beta
        if self.alpha <= 0.0:
            raise ValueError("alpha shold be positive in {}".format(self))
        if self.beta <= 0.0:
            raise ValueError("beta shold be positive in {}".format(self))

    def apply_image(self, img1, img2, factor):
        """ _rand_bbox """
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        cut_rat = np.sqrt(1. - factor)

        cut_w = np.int(w * cut_rat)
        cut_h = np.int(h * cut_rat)

        # uniform
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w - 1)
        bby1 = np.clip(cy - cut_h // 2, 0, h - 1)
        bbx2 = np.clip(cx + cut_w // 2, 0, w - 1)
        bby2 = np.clip(cy + cut_h // 2, 0, h - 1)

        img_1_pad = np.zeros((h, w, img1.shape[2]), 'float32')
        img_1_pad[:img1.shape[0], :img1.shape[1], :] = \
            img1.astype('float32')
        img_2_pad = np.zeros((h, w, img2.shape[2]), 'float32')
        img_2_pad[:img2.shape[0], :img2.shape[1], :] = \
            img2.astype('float32')
        img_1_pad[bby1:bby2, bbx1:bbx2, :] = img_2_pad[bby1:bby2, bbx1:bbx2, :]
        return img_1_pad

    def __call__(self, sample, context=None):
        if not isinstance(sample, Sequence):
            return sample

        assert len(sample) == 2, 'cutmix need two samples'

        factor = np.random.beta(self.alpha, self.beta)
        factor = max(0.0, min(1.0, factor))
        if factor >= 1.0:
            return sample[0]
        if factor <= 0.0:
            return sample[1]
        img1 = sample[0]['image']
        img2 = sample[1]['image']
        img = self.apply_image(img1, img2, factor)
        gt_bbox1 = sample[0]['gt_bbox']
        gt_bbox2 = sample[1]['gt_bbox']
        gt_bbox = np.concatenate((gt_bbox1, gt_bbox2), axis=0)
        gt_class1 = sample[0]['gt_class']
        gt_class2 = sample[1]['gt_class']
        gt_class = np.concatenate((gt_class1, gt_class2), axis=0)
        gt_score1 = np.ones_like(sample[0]['gt_class'])
        gt_score2 = np.ones_like(sample[1]['gt_class'])
        gt_score = np.concatenate(
            (gt_score1 * factor, gt_score2 * (1. - factor)), axis=0)
        result = copy.deepcopy(sample[0])
        result['image'] = img
        result['gt_bbox'] = gt_bbox
        result['gt_score'] = gt_score
        result['gt_class'] = gt_class
        if 'is_crowd' in sample[0]:
            is_crowd1 = sample[0]['is_crowd']
            is_crowd2 = sample[1]['is_crowd']
            is_crowd = np.concatenate((is_crowd1, is_crowd2), axis=0)
            result['is_crowd'] = is_crowd
        if 'difficult' in sample[0]:
            is_difficult1 = sample[0]['difficult']
            is_difficult2 = sample[1]['difficult']
            is_difficult = np.concatenate(
                (is_difficult1, is_difficult2), axis=0)
            result['difficult'] = is_difficult
        return result


# @register_op
class Mixup(BaseOperator):
    """
    Mixup image and gt_bbbox/gt_score
    """

    def __init__(self, alpha=1.5, beta=1.5):
        """ Mixup image and gt_bbbox/gt_score
        Args:
            alpha (float): alpha parameter of beta distribute
            beta (float): beta parameter of beta distribute
        """
        super(Mixup, self).__init__()
        self.alpha = alpha
        self.beta = beta
        if self.alpha <= 0.0:
            raise ValueError("alpha shold be positive in {}".format(self))
        if self.beta <= 0.0:
            raise ValueError("beta shold be positive in {}".format(self))

    def apply_image(self, img1, img2, factor):
        """

        :param img1:
        :param img2:
        :param factor:
        :return:
        """
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        img = np.zeros((h, w, img1.shape[2]), 'float32')
        img[:img1.shape[0], :img1.shape[1], :] = \
            img1.astype('float32') * factor
        img[:img2.shape[0], :img2.shape[1], :] += \
            img2.astype('float32') * (1.0 - factor)
        return img.astype('uint8')

    def __call__(self, sample, context=None):
        if not isinstance(sample, Sequence):
            return sample

        assert len(sample) == 2, 'mixup need two samples'

        factor = np.random.beta(self.alpha, self.beta)
        factor = max(0.0, min(1.0, factor))
        if factor >= 1.0:
            return sample[0]
        if factor <= 0.0:
            return sample[1]
        im = self.apply_image(sample[0]['image'], sample[1]['image'], factor)
        result = copy.deepcopy(sample[0])
        result['image'] = im
        # apply bbox and score
        if 'gt_bbox' in sample[0]:
            gt_bbox1 = sample[0]['gt_bbox']
            gt_bbox2 = sample[1]['gt_bbox']
            gt_bbox = np.concatenate((gt_bbox1, gt_bbox2), axis=0)
            result['gt_bbox'] = gt_bbox
        if 'gt_class' in sample[0]:
            gt_class1 = sample[0]['gt_class']
            gt_class2 = sample[1]['gt_class']
            gt_class = np.concatenate((gt_class1, gt_class2), axis=0)
            result['gt_class'] = gt_class

            gt_score1 = np.ones_like(sample[0]['gt_class'])
            gt_score2 = np.ones_like(sample[1]['gt_class'])
            gt_score = np.concatenate(
                (gt_score1 * factor, gt_score2 * (1. - factor)), axis=0)
            result['gt_score'] = gt_score
        if 'is_crowd' in sample[0]:
            is_crowd1 = sample[0]['is_crowd']
            is_crowd2 = sample[1]['is_crowd']
            is_crowd = np.concatenate((is_crowd1, is_crowd2), axis=0)
            result['is_crowd'] = is_crowd
        if 'difficult' in sample[0]:
            is_difficult1 = sample[0]['difficult']
            is_difficult2 = sample[1]['difficult']
            is_difficult = np.concatenate(
                (is_difficult1, is_difficult2), axis=0)
            result['difficult'] = is_difficult

        return result


from xml.etree import ElementTree as et


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

    l, t, r, b = map(float, [xmin.text, ymin.text, xmax.text, ymax.text])
    return [l, t, r, b]


class Mixup_MergeNew(BaseOperator):
    """
    0,1,2,3 只手
    2x2, 2x3, 3x2, 2x4,4x2, 3x3, 4x4，每个格子放一只手

    Mixup image and gt_bbbox/gt_score
    """

    def __init__(self, pathInfos=None, pathRoot=None, prob=0.5, pathNegs=[]):
        """
        选择标签区域，组成一张新图
        """
        super(Mixup_MergeNew, self).__init__()
        self.allPathInfos = pathInfos
        self.image_dir = pathRoot
        self.prob = prob
        self.negFiles = pathNegs

    def get_img_box_infos(self, line):
        pathimg1, pathxml1 = line.strip().split(" ")
        pathimg = os.path.join(self.image_dir, pathimg1)
        pathxml = os.path.join(self.image_dir, pathxml1)
        img_rgb = cv2.imread(pathimg)[:, :, ::-1]
        gt_bbox = []
        x1, y1, x2, y2 = get_label_box(pathxml)
        gt_bbox.append([x1, y1, x2, y2])

        gt_bbox = np.array(gt_bbox).astype('float32')
        return img_rgb, gt_bbox

    def apply_image(self, img1, img2, factor):
        """
        :param img1:
        :param img2:
        :param factor:
        :return:
        """
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        img = np.zeros((h, w, img1.shape[2]), 'float32')
        img[:img1.shape[0], :img1.shape[1], :] = \
            img1.astype('float32') * factor
        img[:img2.shape[0], :img2.shape[1], :] += \
            img2.astype('float32') * (1.0 - factor)
        return img.astype('uint8')

    def mergeNew(self, list_img, list_box):
        '''
        利用方框截取标签，并拼接成新图像
        '''
        # img_merge = None
        # ggbox_merge = None
        box_list = []
        img0, img1, img2 = list_img
        # img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
        # img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        # img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2BGR)
        box0, box1, box2 = list_box
        box0 = map(int, box0[0])
        box1 = map(int, box1[0])
        box2 = map(int, box2[0])
        # box3 = map(int, box3[0])
        l0, t0, r0, b0 = box0
        l1, t1, r1, b1 = box1
        l2, t2, r2, b2 = box2
        # l3, t3, r3, b3 = box3
        # cv2.rectangle(img0, (l0, t0), (r0, b0), (0, 0, 255), 5)
        # cv2.rectangle(img1, (l1, t1), (r1, b1), (0, 0, 255), 5)
        # cv2.rectangle(img2, (l2, t2), (r2, b2), (0, 0, 255), 5)
        # cv2.rectangle(img3, (l3, t3), (r3, b3), (0, 0, 255), 5)
        # cv2.imwrite("/home/baidu/Desktop/models/model0.jpg", img0)
        # cv2.imwrite("/home/baidu/Desktop/models/model1.jpg", img1)
        # cv2.imwrite("/home/baidu/Desktop/models/model2.jpg", img2)
        # cv2.imwrite("/home/baidu/Desktop/models/model3.jpg", img3)

        fscale = random.uniform(0, 1)

        # 取两张手的数据，每个手640×720,拼接成1280x720的图片
        if fscale < 0.4:
            imgdst = np.ones((720, 1280, 3), dtype=np.uint8)
            h0, w0, _ = img0.shape
            h1, w1, _ = img1.shape

            centx0 = (l0 + r0) / 2.
            centy0 = (t0 + b0) / 2.
            box_w0, box_h0 = r0 - l0, t0 - b0
            box_w0_new = 640
            box_h0_new = 720

            centx1 = (l1 + r1) / 2.
            centy1 = (t1 + b1) / 2.
            box_w1, box_h1 = r1 - l1, t1 - b1
            box_w1_new = 640
            box_h1_new = 720

            if box_w0_new > 640 or box_w1_new > 640:
                img_merge = img0
                ggbox_merge = box0
            else:
                x00_new = max(0, int(centx0 - box_w0_new // 2))
                y00_new = max(0, int(centy0 - box_h0_new // 2))
                x01_new = min(int(x00_new + 640), w0)
                y01_new = min(int(y00_new + 720), h0)
                # print ("new ltrb", x00_new, y00_new, x01_new, y01_new)
                img0_new = img0[y00_new:y01_new, x00_new:x01_new, :]
                l0_new, t0_new, r0_new, b0_new = l0 - x00_new, t0 - y00_new, r0 - x00_new, b0 - y00_new
                imgdst[:(y01_new - y00_new), :(x01_new - x00_new), :] = img0_new
                # cv2.rectangle(imgdst, (l0_new, t0_new), (r0_new, b0_new), (0,0,255), 5, 8, 0)

                x10_new = max(0, int(centx1 - box_w1_new // 2))
                y10_new = max(0, int(centy1 - box_h1_new // 2))
                x11_new = min(int(x10_new + 640), w1)
                y11_new = min(int(y10_new + 720), h1)
                # print("new ltrb", x10_new, y10_new, x11_new, y11_new)
                img1_new = img1[y10_new:y11_new, x10_new:x11_new, :]
                l1_new, t1_new, r1_new, b1_new = l1 - x10_new + 640, t1 - y10_new, r1 - x10_new + 640, b1 - y10_new
                imgdst[:(y11_new - y10_new), 640:(x11_new - x10_new + 640), :] = img1_new
                # cv2.rectangle(imgdst, (l1_new, t1_new), (r1_new, b1_new), (0, 255, 0), 5, 8, 0)
                # cv2.imwrite("/home/baidu/Desktop/models/model"+str(int(time.time()*1000))+".jpg", imgdst)
                box_list.append([l0_new, t0_new, r0_new, b0_new])
                box_list.append([l1_new, t1_new, r1_new, b1_new])


        elif fscale < 0.7:  # 两个手：一个手不变，另一个手随机贴
            imgdst = copy.deepcopy(img0)
            box_list.append([l0, t0, r0, b0])
            h0, w0, _ = img0.shape
            h1, w1, _ = img1.shape
            h2, w2, _ = img1.shape

            centx1 = (l1 + r1) / 2.
            centy1 = (t1 + b1) / 2.
            box_w1, box_h1 = r1 - l1, b1 - t1

            bool_left, bool_right = False, False
            for try_id in range(20):  # 超过最大次数1，则不添加第二只手了
                fratio1 = random.uniform(2, 6)
                box_w1_new, box_h1_new = box_w1 * fratio1, box_h1 * fratio1
                l10 = max(0, int(centx1 - box_w1_new / 2.))
                t10 = max(0, int(centy1 - box_h1_new / 2.))
                r10 = min(int(centx1 + box_w1_new / 2.), w1)
                b10 = min(int(centy1 + box_h1_new / 2.), h1)
                box_w1_new, box_h1_new = r10 - l10, b10 - t10
                # print ("box new", l10, t10, r10, b10)
                if random.uniform(0, 1) < 0.5:  # 判断左边能否放下第二只手
                    if box_w1_new < l0 and box_h1_new < h0:
                        bool_left = True
                        l_start = random.randint(0, l0 - box_w1_new)
                        t_start = random.randint(0, h0 - box_h1_new)
                        imgdst[t_start:t_start + box_h1_new, l_start:l_start + box_w1_new, :] = img1[t10:b10, l10:r10,
                                                                                                :]
                        l1_new = l1 - l10 + l_start
                        t1_new = t1 - t10 + t_start
                        r1_new = r1 - l10 + l_start
                        b1_new = b1 - t10 + t_start
                        # cv2.rectangle(imgdst, (l1_new, t1_new), (r1_new, b1_new), (0, 255, 0), 5, 8, 0)

                else:  # 判断右边能否放下第二只手
                    if w0 - r0 > box_w1_new and box_h1_new < h0:
                        bool_right = True
                        l_start = random.randint(r0, w0 - box_w1_new)
                        t_start = random.randint(0, h0 - box_h1_new)
                        imgdst[t_start:t_start + box_h1_new, l_start:l_start + box_w1_new, :] = img1[t10:b10, l10:r10,
                                                                                                :]
                        l1_new = l1 - l10 + l_start
                        t1_new = t1 - t10 + t_start
                        r1_new = r1 - l10 + l_start
                        b1_new = b1 - t10 + t_start
                        # cv2.rectangle(imgdst, (l1_new, t1_new), (r1_new, b1_new), (0, 255, 0), 5, 8, 0)

                if bool_left or bool_right:
                    # cv2.rectangle(imgdst, (l0, t0), (r0, b0), (0, 0, 255), 5, 8, 0)
                    # cv2.imwrite("/home/baidu/Desktop/models/model" + str(int(time.time() * 1000)) + ".jpg", imgdst)
                    box_list.append([l1_new, t1_new, r1_new, b1_new])
                    break

        else:  # 三个手：一个手不变，另两个手随机贴
            imgdst = copy.deepcopy(img0)
            # cv2.rectangle(imgdst, (l0, t0), (r0, b0), (0,255,255), 3, 8, 0)
            box_list.append([l0, t0, r0, b0])
            h0, w0, _ = img0.shape
            h1, w1, _ = img1.shape
            h2, w2, _ = img2.shape

            bool_left, bool_right = False, False
            for try_id in range(30):  # 超过最大次数1，则不添加第二，三只手了
                if not bool_left:
                    centx1 = (l1 + r1) / 2.
                    centy1 = (t1 + b1) / 2.
                    box_w1, box_h1 = r1 - l1, b1 - t1
                    fratio1 = random.uniform(2, 6)
                    box_w1_new, box_h1_new = box_w1 * fratio1, box_h1 * fratio1
                    l10 = max(0, int(centx1 - box_w1_new / 2.))
                    t10 = max(0, int(centy1 - box_h1_new / 2.))
                    r10 = min(int(centx1 + box_w1_new / 2.), w1)
                    b10 = min(int(centy1 + box_h1_new / 2.), h1)
                    box_w1_new, box_h1_new = r10 - l10, b10 - t10
                    # print ("box new", l10, t10, r10, b10)
                    # 判断左边能否放下第二只手
                    if box_w1_new < l0 and box_h1_new < h0:
                        bool_left = True
                        l_start = random.randint(0, l0 - box_w1_new)
                        t_start = random.randint(0, h0 - box_h1_new)
                        imgdst[t_start:t_start + box_h1_new, l_start:l_start + box_w1_new, :] = img1[t10:b10, l10:r10,
                                                                                                :]
                        l1_new = l1 - l10 + l_start
                        t1_new = t1 - t10 + t_start
                        r1_new = r1 - l10 + l_start
                        b1_new = b1 - t10 + t_start
                        # cv2.rectangle(imgdst, (l1_new, t1_new), (r1_new, b1_new), (0, 255, 0), 5, 8, 0)
                        box_list.append([l1_new, t1_new, r1_new, b1_new])
                if not bool_right:
                    centx2 = (l2 + r2) / 2.
                    centy2 = (t2 + b2) / 2.
                    box_w2, box_h2 = r2 - l2, b2 - t2
                    fratio2 = random.uniform(2, 6)
                    box_w2_new, box_h2_new = box_w2 * fratio2, box_h2 * fratio2
                    l20 = max(0, int(centx2 - box_w2_new / 2.))
                    t20 = max(0, int(centy2 - box_h2_new / 2.))
                    r20 = min(int(centx2 + box_w2_new / 2.), w2)
                    b20 = min(int(centy2 + box_h2_new / 2.), h2)
                    box_w2_new, box_h2_new = r20 - l20, b20 - t20
                    # 判断右边能否放下第三只手
                    if w0 - r0 > box_w2_new and box_h2_new < h0:
                        bool_right = True
                        l_start = random.randint(r0, w0 - box_w2_new)
                        t_start = random.randint(0, h0 - box_h2_new)
                        imgdst[t_start:t_start + box_h2_new, l_start:l_start + box_w2_new, :] = img2[t20:b20, l20:r20,
                                                                                                :]
                        l2_new = l2 - l20 + l_start
                        t2_new = t2 - t20 + t_start
                        r2_new = r2 - l20 + l_start
                        b2_new = b2 - t20 + t_start
                        # cv2.rectangle(imgdst, (l2_new, t2_new), (r2_new, b2_new), (255, 0, 0), 5, 8, 0)
                        box_list.append([l2_new, t2_new, r2_new, b2_new])

                # if bool_left or bool_right:
                #     cv2.imwrite("/home/baidu/Desktop/models/model" + str(int(time.time() * 1000)) + ".jpg", imgdst)

                if bool_left and bool_right:
                    # cv2.rectangle(imgdst, (l0, t0), (r0, b0), (0, 0, 255), 5, 8, 0)
                    # cv2.imwrite("/home/baidu/Desktop/models/model" + str(int(time.time() * 1000)) + ".jpg", imgdst)
                    break

        # 要转为float32, numpy
        # np.clip(box, 0., 1.)
        # img_merge = cv2.cvtColor(img_merge, cv2.COLOR_BGR2RGB)
        img_merge = copy.deepcopy(imgdst)
        ggbox_merge = np.array(box_list).astype(np.float32)
        # ggbox_merge[ggbox_merge[:, 0] < 0] = 0.
        # ggbox_merge[ggbox_merge[:, 0] > imgdst.shape[1]] = imgdst.shape[1]
        # ggbox_merge[ggbox_merge[:, 2] < 0] = 0.
        # ggbox_merge[ggbox_merge[:, 2] > imgdst.shape[1]] = imgdst.shape[1]
        #
        # ggbox_merge[ggbox_merge[:, 1] < 0] = 0.
        # ggbox_merge[ggbox_merge[:, 1] > imgdst.shape[0]] = imgdst.shape[0]
        # ggbox_merge[ggbox_merge[:, 3] < 0] = 0.
        # ggbox_merge[ggbox_merge[:, 3] > imgdst.shape[0]] = imgdst.shape[0]
        # ggbox_merge = np.array(box_list).astype(np.float32)
        # ggbox_merge = np.clip(ggbox_merge, 0.)Mixup_ImgnetMixup_Imgnet

        return img_merge, ggbox_merge

    def __call__(self, sample, context=None):
        result = copy.deepcopy(sample)
        result['only_negimg'] = 0
        # print ("*"*100)
        # print (result)
        # print ("sample", sample)
        f_ratio = random.uniform(0, 1)
        if f_ratio < 0.15:
            return result
        elif f_ratio < 0.4:
            allNums = len(self.allPathInfos)
            id1 = random.randint(0, allNums - 1)
            id2 = random.randint(0, allNums - 1)

            # id3 = random.randint(0, allNums - 1)
            # print ("id123", id1, id2, id3, allNums)
            img1, box1 = self.get_img_box_infos(self.allPathInfos[id1])
            img2, box2 = self.get_img_box_infos(self.allPathInfos[id2])
            # img3, box3 = self.get_img_box_infos(self.allPathInfos[id3])
            img0, box0 = sample['image'], sample['gt_bbox']
            newImage, newBbox = self.mergeNew([img0, img1, img2], [box0, box1, box2])
            result['image'] = newImage
            result["gt_bbox"] = newBbox
            result["gt_score"] = np.ones(shape=(newBbox.shape[0], 1), dtype=np.float32)
            result["gt_class"] = np.zeros(shape=(newBbox.shape[0], 1), dtype=np.float32)
            # cv2.imwrite("/home/baidu/Desktop/models/new_"+str(id)+".jpg", sample['image'])
            return result
        elif f_ratio < 0.9:
            allNums = len(self.allPathInfos)
            img0, box0 = sample['image'], sample['gt_bbox']
            imgsTmp1 = []
            boxTmp1 = []
            imgsTmp1.append(img0)
            boxTmp1.append(box0)
            for i in range(15):
                id_item = random.randint(0, allNums - 1)
                img_item, box_item = self.get_img_box_infos(self.allPathInfos[id_item])
                imgsTmp1.append(img_item)
                boxTmp1.append(box_item)
            newImage, newBbox = operators_MergeNew.mergeImages(imgsTmp1, boxTmp1, self.negFiles)

            # newImage, newBbox = self.mergeNew([img0, img1, img2], [box0, box1, box2])
            result['image'] = newImage
            result["gt_bbox"] = newBbox
            result["gt_score"] = np.ones(shape=(newBbox.shape[0], 1), dtype=np.float32)
            result["gt_class"] = np.zeros(shape=(newBbox.shape[0], 1), dtype=np.float32)
            # for id in range(newBbox.shape[0]):
            #     l,t,r,b = newBbox[id]
            #     l,t,r,b = map(int, [l,t,r,b])
            #     cv2.rectangle(newImage, (l,t), (r,b), (0,255,255), 3, 8, 0)
            # cv2.imwrite("/home/baidu/Desktop/model_img_tmp2/"+str(int(1000*(time.time())))+".jpg", result['image'][:,:,::-1])
            return result
        else:  # 纯背景图
            result1 = copy.deepcopy(result)
            # print ("negFiles")
            nid = random.randint(0, len(self.negFiles) - 1)
            negImg = cv2.imread(self.negFiles[nid])
            h = result['h']
            w = result['w']
            negImg2 = cv2.resize(negImg, (w, h))
            negImg3 = cv2.cvtColor(negImg2, cv2.COLOR_BGR2RGB)
            # for item in result.keys():
            #     if item in ['gt_bbox', 'gt_score', 'gt_class']:
            #         continue
            #     result1[item] = result[item]
            result1['image'] = negImg3
            # result1['gt_bbox'] = None
            # result1['gt_score'] = None
            # result1['gt_class'] = None
            result1['only_negimg'] = 1
            return result1


class Mixup_MergeNew_uniform(BaseOperator):
    """
    Mixup image and gt_bbbox/gt_score
    """

    def __init__(self, pathInfos=None, pathRoot=None, prob=0.5, pathNegs=[]):
        """
        选择标签区域，组成一张新图
        """
        super(Mixup_MergeNew_uniform, self).__init__()
        self.allPathInfos = pathInfos
        self.image_dir = pathRoot
        self.prob = prob
        self.negFiles = pathNegs

    def get_img_box_infos(self, line):
        pathimg1, pathxml1 = line.strip().split(" ")
        pathimg = os.path.join(self.image_dir, pathimg1)
        pathxml = os.path.join(self.image_dir, pathxml1)
        img_rgb = cv2.imread(pathimg)[:, :, ::-1]
        gt_bbox = []
        x1, y1, x2, y2 = get_label_box(pathxml)
        gt_bbox.append([x1, y1, x2, y2])

        gt_bbox = np.array(gt_bbox).astype('float32')
        return img_rgb, gt_bbox

    def apply_image(self, img1, img2, factor):
        """
        :param img1:
        :param img2:
        :param factor:
        :return:
        """
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        img = np.zeros((h, w, img1.shape[2]), 'float32')
        img[:img1.shape[0], :img1.shape[1], :] = \
            img1.astype('float32') * factor
        img[:img2.shape[0], :img2.shape[1], :] += \
            img2.astype('float32') * (1.0 - factor)
        return img.astype('uint8')

    def mergeNew(self, list_img, list_box):
        '''
        利用方框截取标签，并拼接成新图像
        '''
        # img_merge = None
        # ggbox_merge = None
        box_list = []
        img0, img1, img2 = list_img
        # img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
        # img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        # img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2BGR)
        box0, box1, box2 = list_box
        box0 = map(int, box0[0])
        box1 = map(int, box1[0])
        box2 = map(int, box2[0])
        # box3 = map(int, box3[0])
        l0, t0, r0, b0 = box0
        l1, t1, r1, b1 = box1
        l2, t2, r2, b2 = box2
        # l3, t3, r3, b3 = box3
        # cv2.rectangle(img0, (l0, t0), (r0, b0), (0, 0, 255), 5)
        # cv2.rectangle(img1, (l1, t1), (r1, b1), (0, 0, 255), 5)
        # cv2.rectangle(img2, (l2, t2), (r2, b2), (0, 0, 255), 5)
        # cv2.rectangle(img3, (l3, t3), (r3, b3), (0, 0, 255), 5)
        # cv2.imwrite("/home/baidu/Desktop/models/model0.jpg", img0)
        # cv2.imwrite("/home/baidu/Desktop/models/model1.jpg", img1)
        # cv2.imwrite("/home/baidu/Desktop/models/model2.jpg", img2)
        # cv2.imwrite("/home/baidu/Desktop/models/model3.jpg", img3)

        fscale = random.uniform(0, 1)

        # 取两张手的数据，每个手640×720,拼接成1280x720的图片
        if fscale < 0.4:
            imgdst = np.ones((720, 1280, 3), dtype=np.uint8)
            h0, w0, _ = img0.shape
            h1, w1, _ = img1.shape

            centx0 = (l0 + r0) / 2.
            centy0 = (t0 + b0) / 2.
            box_w0, box_h0 = r0 - l0, t0 - b0
            box_w0_new = 640
            box_h0_new = 720

            centx1 = (l1 + r1) / 2.
            centy1 = (t1 + b1) / 2.
            box_w1, box_h1 = r1 - l1, t1 - b1
            box_w1_new = 640
            box_h1_new = 720

            if box_w0_new > 640 or box_w1_new > 640:
                img_merge = img0
                ggbox_merge = box0
            else:
                x00_new = max(0, int(centx0 - box_w0_new // 2))
                y00_new = max(0, int(centy0 - box_h0_new // 2))
                x01_new = min(int(x00_new + 640), w0)
                y01_new = min(int(y00_new + 720), h0)
                # print ("new ltrb", x00_new, y00_new, x01_new, y01_new)
                img0_new = img0[y00_new:y01_new, x00_new:x01_new, :]
                l0_new, t0_new, r0_new, b0_new = l0 - x00_new, t0 - y00_new, r0 - x00_new, b0 - y00_new
                imgdst[:(y01_new - y00_new), :(x01_new - x00_new), :] = img0_new
                # cv2.rectangle(imgdst, (l0_new, t0_new), (r0_new, b0_new), (0,0,255), 5, 8, 0)

                x10_new = max(0, int(centx1 - box_w1_new // 2))
                y10_new = max(0, int(centy1 - box_h1_new // 2))
                x11_new = min(int(x10_new + 640), w1)
                y11_new = min(int(y10_new + 720), h1)
                # print("new ltrb", x10_new, y10_new, x11_new, y11_new)
                img1_new = img1[y10_new:y11_new, x10_new:x11_new, :]
                l1_new, t1_new, r1_new, b1_new = l1 - x10_new + 640, t1 - y10_new, r1 - x10_new + 640, b1 - y10_new
                imgdst[:(y11_new - y10_new), 640:(x11_new - x10_new + 640), :] = img1_new
                # cv2.rectangle(imgdst, (l1_new, t1_new), (r1_new, b1_new), (0, 255, 0), 5, 8, 0)
                # cv2.imwrite("/home/baidu/Desktop/models/model"+str(int(time.time()*1000))+".jpg", imgdst)
                box_list.append([l0_new, t0_new, r0_new, b0_new])
                box_list.append([l1_new, t1_new, r1_new, b1_new])


        elif fscale < 0.7:  # 两个手：一个手不变，另一个手随机贴
            imgdst = copy.deepcopy(img0)
            box_list.append([l0, t0, r0, b0])
            h0, w0, _ = img0.shape
            h1, w1, _ = img1.shape
            h2, w2, _ = img1.shape

            centx1 = (l1 + r1) / 2.
            centy1 = (t1 + b1) / 2.
            box_w1, box_h1 = r1 - l1, b1 - t1

            bool_left, bool_right = False, False
            for try_id in range(20):  # 超过最大次数1，则不添加第二只手了
                fratio1 = random.uniform(2, 6)
                box_w1_new, box_h1_new = box_w1 * fratio1, box_h1 * fratio1
                l10 = max(0, int(centx1 - box_w1_new / 2.))
                t10 = max(0, int(centy1 - box_h1_new / 2.))
                r10 = min(int(centx1 + box_w1_new / 2.), w1)
                b10 = min(int(centy1 + box_h1_new / 2.), h1)
                box_w1_new, box_h1_new = r10 - l10, b10 - t10
                # print ("box new", l10, t10, r10, b10)
                if random.uniform(0, 1) < 0.5:  # 判断左边能否放下第二只手
                    if box_w1_new < l0 and box_h1_new < h0:
                        bool_left = True
                        l_start = random.randint(0, l0 - box_w1_new)
                        t_start = random.randint(0, h0 - box_h1_new)
                        imgdst[t_start:t_start + box_h1_new, l_start:l_start + box_w1_new, :] = img1[t10:b10, l10:r10,
                                                                                                :]
                        l1_new = l1 - l10 + l_start
                        t1_new = t1 - t10 + t_start
                        r1_new = r1 - l10 + l_start
                        b1_new = b1 - t10 + t_start
                        # cv2.rectangle(imgdst, (l1_new, t1_new), (r1_new, b1_new), (0, 255, 0), 5, 8, 0)

                else:  # 判断右边能否放下第二只手
                    if w0 - r0 > box_w1_new and box_h1_new < h0:
                        bool_right = True
                        l_start = random.randint(r0, w0 - box_w1_new)
                        t_start = random.randint(0, h0 - box_h1_new)
                        imgdst[t_start:t_start + box_h1_new, l_start:l_start + box_w1_new, :] = img1[t10:b10, l10:r10,
                                                                                                :]
                        l1_new = l1 - l10 + l_start
                        t1_new = t1 - t10 + t_start
                        r1_new = r1 - l10 + l_start
                        b1_new = b1 - t10 + t_start
                        # cv2.rectangle(imgdst, (l1_new, t1_new), (r1_new, b1_new), (0, 255, 0), 5, 8, 0)

                if bool_left or bool_right:
                    # cv2.rectangle(imgdst, (l0, t0), (r0, b0), (0, 0, 255), 5, 8, 0)
                    # cv2.imwrite("/home/baidu/Desktop/models/model" + str(int(time.time() * 1000)) + ".jpg", imgdst)
                    box_list.append([l1_new, t1_new, r1_new, b1_new])
                    break

        else:  # 三个手：一个手不变，另两个手随机贴
            imgdst = copy.deepcopy(img0)
            # cv2.rectangle(imgdst, (l0, t0), (r0, b0), (0,255,255), 3, 8, 0)
            box_list.append([l0, t0, r0, b0])
            h0, w0, _ = img0.shape
            h1, w1, _ = img1.shape
            h2, w2, _ = img2.shape

            bool_left, bool_right = False, False
            for try_id in range(30):  # 超过最大次数1，则不添加第二，三只手了
                if not bool_left:
                    centx1 = (l1 + r1) / 2.
                    centy1 = (t1 + b1) / 2.
                    box_w1, box_h1 = r1 - l1, b1 - t1
                    fratio1 = random.uniform(2, 6)
                    box_w1_new, box_h1_new = box_w1 * fratio1, box_h1 * fratio1
                    l10 = max(0, int(centx1 - box_w1_new / 2.))
                    t10 = max(0, int(centy1 - box_h1_new / 2.))
                    r10 = min(int(centx1 + box_w1_new / 2.), w1)
                    b10 = min(int(centy1 + box_h1_new / 2.), h1)
                    box_w1_new, box_h1_new = r10 - l10, b10 - t10
                    # print ("box new", l10, t10, r10, b10)
                    # 判断左边能否放下第二只手
                    if box_w1_new < l0 and box_h1_new < h0:
                        bool_left = True
                        l_start = random.randint(0, l0 - box_w1_new)
                        t_start = random.randint(0, h0 - box_h1_new)
                        imgdst[t_start:t_start + box_h1_new, l_start:l_start + box_w1_new, :] = img1[t10:b10, l10:r10,
                                                                                                :]
                        l1_new = l1 - l10 + l_start
                        t1_new = t1 - t10 + t_start
                        r1_new = r1 - l10 + l_start
                        b1_new = b1 - t10 + t_start
                        # cv2.rectangle(imgdst, (l1_new, t1_new), (r1_new, b1_new), (0, 255, 0), 5, 8, 0)
                        box_list.append([l1_new, t1_new, r1_new, b1_new])
                if not bool_right:
                    centx2 = (l2 + r2) / 2.
                    centy2 = (t2 + b2) / 2.
                    box_w2, box_h2 = r2 - l2, b2 - t2
                    fratio2 = random.uniform(2, 6)
                    box_w2_new, box_h2_new = box_w2 * fratio2, box_h2 * fratio2
                    l20 = max(0, int(centx2 - box_w2_new / 2.))
                    t20 = max(0, int(centy2 - box_h2_new / 2.))
                    r20 = min(int(centx2 + box_w2_new / 2.), w2)
                    b20 = min(int(centy2 + box_h2_new / 2.), h2)
                    box_w2_new, box_h2_new = r20 - l20, b20 - t20
                    # 判断右边能否放下第三只手
                    if w0 - r0 > box_w2_new and box_h2_new < h0:
                        bool_right = True
                        l_start = random.randint(r0, w0 - box_w2_new)
                        t_start = random.randint(0, h0 - box_h2_new)
                        imgdst[t_start:t_start + box_h2_new, l_start:l_start + box_w2_new, :] = img2[t20:b20, l20:r20,
                                                                                                :]
                        l2_new = l2 - l20 + l_start
                        t2_new = t2 - t20 + t_start
                        r2_new = r2 - l20 + l_start
                        b2_new = b2 - t20 + t_start
                        # cv2.rectangle(imgdst, (l2_new, t2_new), (r2_new, b2_new), (255, 0, 0), 5, 8, 0)
                        box_list.append([l2_new, t2_new, r2_new, b2_new])

                # if bool_left or bool_right:
                #     cv2.imwrite("/home/baidu/Desktop/models/model" + str(int(time.time() * 1000)) + ".jpg", imgdst)

                if bool_left and bool_right:
                    # cv2.rectangle(imgdst, (l0, t0), (r0, b0), (0, 0, 255), 5, 8, 0)
                    # cv2.imwrite("/home/baidu/Desktop/models/model" + str(int(time.time() * 1000)) + ".jpg", imgdst)
                    break

        # 要转为float32, numpy
        # np.clip(box, 0., 1.)
        # img_merge = cv2.cvtColor(img_merge, cv2.COLOR_BGR2RGB)
        img_merge = copy.deepcopy(imgdst)
        ggbox_merge = np.array(box_list).astype(np.float32)
        # ggbox_merge[ggbox_merge[:, 0] < 0] = 0.
        # ggbox_merge[ggbox_merge[:, 0] > imgdst.shape[1]] = imgdst.shape[1]
        # ggbox_merge[ggbox_merge[:, 2] < 0] = 0.
        # ggbox_merge[ggbox_merge[:, 2] > imgdst.shape[1]] = imgdst.shape[1]
        #
        # ggbox_merge[ggbox_merge[:, 1] < 0] = 0.
        # ggbox_merge[ggbox_merge[:, 1] > imgdst.shape[0]] = imgdst.shape[0]
        # ggbox_merge[ggbox_merge[:, 3] < 0] = 0.
        # ggbox_merge[ggbox_merge[:, 3] > imgdst.shape[0]] = imgdst.shape[0]
        # ggbox_merge = np.array(box_list).astype(np.float32)
        # ggbox_merge = np.clip(ggbox_merge, 0.)Mixup_ImgnetMixup_Imgnet

        return img_merge, ggbox_merge

    def mergeNew_2_6Hands(self, list_img, list_box, allNegFiles):
        '''
        利用方框截取标签，并拼接成新图像
        '''
        norm_h = 720
        norm_w = 1280
        try:
            # 选取一张图片作为背景图片
            rand_neg_id = random.randint(0, len(allNegFiles) - 1)
            imgdst = cv2.imread(allNegFiles[rand_neg_id])
            imgdst = cv2.cvtColor(imgdst, cv2.COLOR_BGR2RGB)
            imgdst = cv2.resize(imgdst, (norm_w, norm_h))
        except:
            imgdst = np.ones((norm_h, norm_w, 3), dtype=np.uint8)
        # to do: 正样本图片遮挡手作为背景
        box_dst = []

        # img_merge = None
        # ggbox_merge = None
        # box_list = []
        img0, img1, img2, img3, img4, img5 = list_img
        # img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
        # img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        # img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2BGR)
        # img4 = cv2.cvtColor(img4, cv2.COLOR_RGB2BGR)
        # img5 = cv2.cvtColor(img5, cv2.COLOR_RGB2BGR)

        list_box1 = copy.deepcopy(list_box)
        list_box = []
        for item in list_box1:
            list_box.append(list(item[0]))

        # box0, box1, box2, box3, box4, box5 = list_box
        # box0 = map(int, box0)
        # box1 = map(int, box1)
        # box2 = map(int, box2)
        # box3 = map(int, box3)
        # box4 = map(int, box4)
        # box5 = map(int, box5)
        # l0, t0, r0, b0 = box0
        # l1, t1, r1, b1 = box1
        # l2, t2, r2, b2 = box2
        # l3, t3, r3, b3 = box3
        # l4, t4, r4, b4 = box4
        # l5, t5, r5, b5 = box5
        # cv2.rectangle(img0, (l0, t0), (r0, b0), (0, 0, 255), 5)
        # cv2.rectangle(img1, (l1, t1), (r1, b1), (0, 0, 255), 5)
        # cv2.rectangle(img2, (l2, t2), (r2, b2), (0, 0, 255), 5)
        # cv2.rectangle(img3, (l3, t3), (r3, b3), (0, 0, 255), 5)
        # cv2.rectangle(img4, (l4, t4), (r4, b4), (0, 0, 255), 5)
        # cv2.rectangle(img5, (l5, t5), (r5, b5), (0, 0, 255), 5)
        # cv2.imwrite("/media/baidu/ssd2/ppyolo/6w_data/small_test/less_handNum6/testImg/model0.jpg", img0)
        # cv2.imwrite("/media/baidu/ssd2/ppyolo/6w_data/small_test/less_handNum6/testImg/model1.jpg", img1)
        # cv2.imwrite("/media/baidu/ssd2/ppyolo/6w_data/small_test/less_handNum6/testImg/model2.jpg", img2)
        # cv2.imwrite("/media/baidu/ssd2/ppyolo/6w_data/small_test/less_handNum6/testImg/model3.jpg", img3)
        # cv2.imwrite("/media/baidu/ssd2/ppyolo/6w_data/small_test/less_handNum6/testImg/model4.jpg", img4)
        # cv2.imwrite("/media/baidu/ssd2/ppyolo/6w_data/small_test/less_handNum6/testImg/model5.jpg", img5)

        num_hand_choice = random.choice([1, 2, 3, 4, 5, 6])
        paste_hand_num = 0
        id_img = 0
        id_box = 0
        for i in range(300):  # 最多尝试300次
            img_item = copy.deepcopy(list_img[id_img])
            img_h, img_w, img_c = img_item.shape
            box_item = list_box[id_box]
            # box_item = box_item
            l, t, r, b = box_item
            centx = (l + r) / 2.
            centy = (t + b) / 2.
            box_w = r - l
            box_h = b - t
            expand_ratio = random.uniform(1.2, 4)
            l_new = max(0, centx - box_w * expand_ratio * 0.5)
            r_new = min(img_w - 1, centx + box_w * expand_ratio * 0.5)
            t_new = max(0, centy - box_h * expand_ratio * 0.5)
            b_new = min(img_h - 1, centy + box_h * expand_ratio * 0.5)
            l_new, t_new, r_new, b_new = map(int, [l_new, t_new, r_new, b_new])
            img_hand = img_item[t_new:b_new, l_new:r_new, :]
            # print ("img_hand,img_item", img_item.shape, img_hand.shape)
            # cv2.imwrite("/media/baidu/ssd2/ppyolo/6w_data/small_test/less_handNum6/testHand/" + str(i).zfill(5) + "_img_hand.jpg", img_hand[:,:,::-1])
            img_hand_h, img_hand_w, img_hand_c = img_hand.shape
            # 截取的手的图像上的box
            l_tmp1 = l - l_new
            r_tmp1 = r - l_new
            t_tmp1 = t - t_new
            b_tmp1 = b - t_new
            if img_hand_w >= norm_w or img_hand_h > norm_h:
                continue
            rand_x = random.randint(0, norm_w - img_hand_w)
            rand_y = random.randint(0, norm_h - img_hand_h)
            # 如果贴到目标图上，手的座标变为
            hand_l = rand_x + l_tmp1
            hand_r = rand_x + r_tmp1
            hand_t = rand_y + t_tmp1
            hand_b = rand_y + b_tmp1
            # box的iou参考意义u不大
            # box_iou = bbox_utils.compute_boxIOU([hand_l, hand_t, hand_r, hand_b], box_dst)
            # 计算新图是否遮挡了之前存在的手，如果被遮挡的面积大于等于0.1,则不可粘贴
            imgwh_box_iou = bbox_utils.compute_imgwh_boxIOU([rand_x, rand_y, rand_x + img_hand_w, rand_y + img_hand_h],
                                                            box_dst)
            if imgwh_box_iou < 0.1:
                imgdst[rand_y:rand_y + img_hand_h, rand_x:rand_x + img_hand_w, :] = copy.deepcopy(img_hand)
                box_dst.append([hand_l, hand_t, hand_r, hand_b])
                paste_hand_num += 1
                id_img += 1
                id_box += 1
                if paste_hand_num == num_hand_choice:
                    break

            # imgdst_tmp = copy.deepcopy(imgdst)
            # imgdst_tmp[rand_y:rand_y+img_hand_h, rand_x:rand_x+img_hand_w, : ] = img_hand

        if len(box_dst) == 0:
            # print ("len_box_dst=0", list_img[0].shape, type(list_box[0]), list_box[0])
            return list_img[0], np.array([list_box[0]]).astype(np.float32)
        else:
            # imgdst_tmp = copy.deepcopy(imgdst)
            # print ("imgdst_boxdst", imgdst_tmp.shape, box_dst)
            # print ("boxdst_len", len(box_dst))
            # for box_tmp in box_dst:
            #     l, t, r, b = map(int, box_tmp)
            #     cv2.rectangle(imgdst_tmp, (l, t), (r, b), (0, 0, 255), 5, 8, 0)
            # cv2.imwrite("/media/baidu/ssd2/ppyolo/6w_data/small_test/less_handNum6/testHand/" + str(i).zfill(5) +str(int(1000*time.time()))+ ".jpg", imgdst_tmp[:,:,::-1])
            return imgdst, np.array(box_dst).astype(np.float32)

    def __call__(self, sample, context=None):
        result = copy.deepcopy(sample)
        result['only_negimg'] = 0
        result['crop_MaxHand'] = 0
        # print ("*"*100)
        # print (result)
        # print ("sample", sample)
        f_ratio = random.uniform(0, 1)
        if f_ratio < 0.1:
            result['crop_MaxHand'] = 1
            return result
        # 2,3,4,5,6只手
        elif f_ratio < 0.85:
            allNums = len(self.allPathInfos)
            img0, box0 = sample['image'], sample['gt_bbox']
            imgsTmp1 = []
            boxTmp1 = []
            imgsTmp1.append(img0)
            boxTmp1.append(box0)
            for i in range(5):
                id_item = random.randint(0, allNums - 1)
                img_item, box_item = self.get_img_box_infos(self.allPathInfos[id_item])
                imgsTmp1.append(img_item)
                boxTmp1.append(box_item)
            # newImage, newBbox = operators_MergeNew.mergeImages(imgsTmp1, boxTmp1, self.negFiles)

            newImage, newBbox = self.mergeNew_2_6Hands(imgsTmp1, boxTmp1, self.negFiles)
            # print ("nweImgBox", newImage.shape, newBbox.shape, newBbox)
            # cv2.imwrite("/media/baidu/ssd2/ppyolo/6w_data/small_test/less_handNum6/newImage/tmp.jpg", newImage)
            # newImage, newBbox = self.mergeNew([img0, img1, img2], [box0, box1, box2])
            # return result
            result['image'] = newImage
            result["gt_bbox"] = newBbox
            result["gt_score"] = np.ones(shape=(newBbox.shape[0], 1), dtype=np.float32)
            result["gt_class"] = np.zeros(shape=(newBbox.shape[0], 1), dtype=np.float32)
            return result
        #
        # elif f_ratio < 0.4:  #2,3只手
        #     allNums = len(self.allPathInfos)
        #     id1 = random.randint(0, allNums - 1)
        #     id2 = random.randint(0, allNums - 1)
        #
        #     # id3 = random.randint(0, allNums - 1)
        #     # print ("id123", id1, id2, id3, allNums)
        #     img1, box1 = self.get_img_box_infos(self.allPathInfos[id1])
        #     img2, box2 = self.get_img_box_infos(self.allPathInfos[id2])
        #     # img3, box3 = self.get_img_box_infos(self.allPathInfos[id3])
        #     img0, box0 = sample['image'], sample['gt_bbox']
        #     newImage, newBbox = self.mergeNew([img0, img1, img2], [box0, box1, box2])
        #     result['image'] = newImage
        #     result["gt_bbox"] = newBbox
        #     result["gt_score"] = np.ones(shape=(newBbox.shape[0], 1), dtype=np.float32)
        #     result["gt_class"] = np.zeros(shape=(newBbox.shape[0], 1), dtype=np.float32)
        #     # cv2.imwrite("/home/baidu/Desktop/models/new_"+str(id)+".jpg", sample['image'])
        #     return result
        # elif f_ratio < 0.9: #4,6,6,8,8,9,16只手
        #     allNums = len(self.allPathInfos)
        #     img0, box0 = sample['image'], sample['gt_bbox']
        #     imgsTmp1 = []
        #     boxTmp1 = []
        #     imgsTmp1.append(img0)
        #     boxTmp1.append(box0)
        #     for i in range(15):
        #         id_item = random.randint(0, allNums - 1)
        #         img_item, box_item = self.get_img_box_infos(self.allPathInfos[id_item])
        #         imgsTmp1.append(img_item)
        #         boxTmp1.append(box_item)
        #     newImage, newBbox = operators_MergeNew.mergeImages(imgsTmp1, boxTmp1, self.negFiles)
        #
        #     # newImage, newBbox = self.mergeNew([img0, img1, img2], [box0, box1, box2])
        #     result['image'] = newImage
        #     result["gt_bbox"] = newBbox
        #     result["gt_score"] = np.ones(shape=(newBbox.shape[0], 1), dtype=np.float32)
        #     result["gt_class"] = np.zeros(shape=(newBbox.shape[0], 1), dtype=np.float32)
        #     # for id in range(newBbox.shape[0]):
        #     #     l,t,r,b = newBbox[id]
        #     #     l,t,r,b = map(int, [l,t,r,b])
        #     #     cv2.rectangle(newImage, (l,t), (r,b), (0,255,255), 3, 8, 0)
        #     # cv2.imwrite("/home/baidu/Desktop/model_img_tmp2/"+str(int(1000*(time.time())))+".jpg", result['image'][:,:,::-1])
        #     return result
        else:  # 纯背景图
            result1 = copy.deepcopy(result)
            # print ("negFiles")
            nid = random.randint(0, len(self.negFiles) - 1)
            negImg = cv2.imread(self.negFiles[nid])
            h = result['h']
            w = result['w']
            negImg2 = cv2.resize(negImg, (w, h))
            negImg3 = cv2.cvtColor(negImg2, cv2.COLOR_BGR2RGB)
            # for item in result.keys():
            #     if item in ['gt_bbox', 'gt_score', 'gt_class']:
            #         continue
            #     result1[item] = result[item]
            result1['image'] = negImg3
            # result1['gt_bbox'] = None
            # result1['gt_score'] = None
            # result1['gt_class'] = None
            result1['only_negimg'] = 1
            return result1


# 操作人脸，组成1,2,3,4,5张人脸，
class Mixup_MergeNewFace(BaseOperator):
    """
    0, 20% （人脸方框遮挡 10%， imgnet无人脸 10%, #to do： coco无人脸）
    1, 20%
    2,3 个人脸 15%,15%
    4,5个人脸  15%,15%

    Mixup image and gt_bbbox/gt_score
    """

    def __init__(self, pathInfos=None, pathRoot=None, prob=0.5, pathNegs=[]):
        """
        选择标签区域，组成一张新图
        """
        super(Mixup_MergeNewFace, self).__init__()
        self.allPathInfos = pathInfos
        self.image_dir = pathRoot
        self.prob = prob
        self.negFiles = pathNegs

    def get_img_box_infos(self, line):
        pathimg1, pathxml1 = line.strip().split(" ")
        pathimg = os.path.join(self.image_dir, pathimg1)
        pathxml = os.path.join(self.image_dir, pathxml1)
        img_rgb = cv2.imread(pathimg)[:, :, ::-1]
        gt_bbox = []
        x1, y1, x2, y2 = get_label_box(pathxml)
        gt_bbox.append([x1, y1, x2, y2])

        gt_bbox = np.array(gt_bbox).astype('float32')
        return img_rgb, gt_bbox

    def get_img_box_points(self, pathImg):
        def random_rotate(img, pts, frand_angle=60):
            # print ("rotate",img.shape, pts.shape)
            h, w = img.shape
            cx = w / 2
            cy = h / 2
            angle = random.uniform(-frand_angle, frand_angle)
            # angle = random.uniform(40, 80)
            rotate_m = cv.getRotationMatrix2D((cx, cy), angle, 1.0)
            rotate_cos = np.abs(rotate_m[0, 0])
            rotate_sin = np.abs(rotate_m[0, 1])
            # # compute the new bounding dimensions of the image
            new_width = int((h * rotate_sin) + (w * rotate_cos))
            new_height = int((h * rotate_cos) + (w * rotate_sin))

            # # adjust the rotation matrix to take into account translation
            rotate_m[0, 2] += (new_width / 2) - cx
            rotate_m[1, 2] += (new_height / 2) - cy
            # new_image = cv.warpAffine(img, rotate_m, (new_width, new_height), borderValue=(255, 255, 255))
            new_image = cv.warpAffine(img, rotate_m, (new_width, new_height), borderValue=128)
            new_image2 = copy.deepcopy(new_image)

            R = rotate_m[0:2, 0:2]
            T = rotate_m[:, 2]
            new_facial_landmark = np.dot(R, pts.T).T + T
            for i in range(new_facial_landmark.shape[0]):
                x,y = new_facial_landmark[i]
                x,y = int(x),int(y)
                cv.circle(new_image2, (x,y), 3, (0,0,255), -1, 8, 0)
            # cv.imwrite("/root/paddlejob/data_train/train_mid_img/test.jpg", new_image2)
            return new_image, new_facial_landmark


        pathImg = pathImg.strip()
        label_infos = {}
        label_infos['facebox'] = []
        label_infos['facepoints'] = []

        imgbgr = cv.imread(os.path.join(self.image_dir, pathImg))
        # imgrgb = cv.cvtColor(imgbgr, cv.COLOR_BGR2RGB)
        imggray = cv.cvtColor(imgbgr, cv.COLOR_BGR2GRAY)
        try:
            filepathInfo = pathImg.split("/")
            imgname = filepathInfo[-1]
            imgname1 = imgname.split(".")[:-1]
            imgname2 = ".".join(imgname1)
            filepathPrefix = "/".join(filepathInfo[:-1])
            if "/1000personFacialLandmark/标2_ModifyTop/" in pathImg or "oms_SmallFace_biaozhu" in pathImg:
                jsonname = imgname2 + ".json"
            else:
                jsonname = "out_" + imgname2 + ".json"
            pathJson1 = os.path.join(filepathPrefix, jsonname)
            pathJson = os.path.join(self.image_dir, pathJson1)
            infos = json.loads(open(pathJson).read())
            WorkLoad = infos['WorkLoad']
            DataList = infos["DataList"]
            Point_num = WorkLoad['Point Num']
            fscale = WorkLoad["scale_x"]
            # 目前Point_num 有三个数值，2, 106, 72,
            # 暂时先设置一个阈值5,大于等于5则使用人脸点推理方框，否则直接读取face_bbox

            if Point_num >= 5:
                allxy = []
                for item in DataList:
                    if item['type'] != "Point":
                        continue
                    fx, fy = item['coordinates']
                    # label_infos['facepoints'].append([fx, fy])
                    allxy.append([fx*fscale, fy*fscale])
                allxy = np.array(allxy)
                newimg, newpt = random_rotate(imggray, allxy)
                # all_x = []
                # all_y = []
                # left = min(all_x)
                # right = max(all_x)
                # top = min(all_y)
                # bottom = max(all_y)
                left = np.min(newpt[:, 0])
                right = np.max(newpt[:, 0])
                top = np.min(newpt[:, 1])
                bottom = np.max(newpt[:, 1])
                fscale = 1.
                imggray = newimg
            else:
                for item in DataList:
                    if item['type'] == "face_bbox":
                        coordinates = item['coordinates']
                        left = coordinates[0]['left']
                        top = coordinates[1]['top']
                        right = coordinates[2]['right']
                        bottom = coordinates[3]['bottom']
                        label_infos['facebox'] = [left, top, right, bottom]
            if right - left < 10 or bottom - top < 10:
                return imggray, []
            return imggray, [left * fscale, top * fscale, right * fscale, bottom * fscale]
        except:
            # print("pathJson error:", pathJson)
            return imggray, []

    def mergeNew_5Faces(self, list_img, list_box, allNegFiles):
        '''
        利用方框截取标签，并拼接成新图像
        '''
        norm_h = int(360 * 1.5)
        norm_w = int(640 * 1.5)
        try:
            # 选取一张图片作为背景图片
            rand_neg_id = random.randint(0, len(allNegFiles) - 1)
            imgdst = cv2.imread(allNegFiles[rand_neg_id], 0)
            # imgdst = cv2.cvtColor(imgdst, cv2.COLOR_BGR2GRAY)
            imgdst = cv2.resize(imgdst, (norm_w, norm_h))
        except:
            imgdst = np.ones((norm_h, norm_w), dtype=np.uint8)
        # to do: 正样本图片遮挡脸作为背景
        box_dst = []

        # img_merge = None
        # ggbox_merge = None
        # box_list = []
        img0, img1, img2, img3, img4 = list_img
        # img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
        # img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        # img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2BGR)
        # img4 = cv2.cvtColor(img4, cv2.COLOR_RGB2BGR)

        list_box[0] = list(list_box[0][0])
        # list_box1 = copy.deepcopy(list_box)
        # list_box = []
        # for item in list_box1:
        #     print ("item:", item)
        #     list_box.append(list(item[0]))

        # box0, box1, box2, box3, box4 = list_box
        # box0 = map(int, box0)
        # box1 = map(int, box1)
        # box2 = map(int, box2)
        # box3 = map(int, box3)
        # box4 = map(int, box4)

        # l0, t0, r0, b0 = box0
        # l1, t1, r1, b1 = box1
        # l2, t2, r2, b2 = box2
        # l3, t3, r3, b3 = box3
        # l4, t4, r4, b4 = box4

        # cv2.rectangle(img0, (l0, t0), (r0, b0), (0, 0, 255), 5)
        # cv2.rectangle(img1, (l1, t1), (r1, b1), (0, 0, 255), 5)
        # cv2.rectangle(img2, (l2, t2), (r2, b2), (0, 0, 255), 5)
        # cv2.rectangle(img3, (l3, t3), (r3, b3), (0, 0, 255), 5)
        # cv2.rectangle(img4, (l4, t4), (r4, b4), (0, 0, 255), 5)

        # cv2.imwrite("/home/baidu/Desktop/6faces/model0.jpg", img0)
        # cv2.imwrite("/home/baidu/Desktop/6faces/model1.jpg", img1)
        # cv2.imwrite("/home/baidu/Desktop/6faces/model2.jpg", img2)
        # cv2.imwrite("/home/baidu/Desktop/6faces/model3.jpg", img3)
        # cv2.imwrite("/home/baidu/Desktop/6faces/model4.jpg", img4)

        num_face_choice = np.random.choice([1, 2, 3, 4, 5], p=[0, 0, 0, 0.5, 0.5])
        paste_face_num = 0
        for try_num in range(3):
            if paste_face_num == num_face_choice:
                break
            paste_face_num = 0
            id_img = 0
            id_box = 0
            imgdst_try = copy.deepcopy(imgdst)
            box_dst_try = copy.deepcopy(box_dst)
            for i in range(60):  # 最多尝试100次
                img_item = copy.deepcopy(list_img[id_img])
                img_h, img_w = img_item.shape
                box_item = list_box[id_box]
                # box_item = box_item
                l, t, r, b = box_item
                centx = (l + r) / 2.
                centy = (t + b) / 2.
                box_w = r - l
                box_h = b - t
                expand_ratio = random.uniform(2, 4)
                l_new = max(0, centx - box_w * expand_ratio * 0.5)
                r_new = min(img_w - 1, centx + box_w * expand_ratio * 0.5)
                t_new = max(0, centy - box_h * expand_ratio * 0.5)
                b_new = min(img_h - 1, centy + box_h * expand_ratio * 0.5)
                l_new, t_new, r_new, b_new = map(int, [l_new, t_new, r_new, b_new])
                img_face = img_item[t_new:b_new, l_new:r_new]
                # print ("img_hand,img_item", img_item.shape, img_hand.shape)
                # cv2.imwrite("/media/baidu/ssd2/ppyolo/6w_data/small_test/less_handNum6/testHand/" + str(i).zfill(5) + "_img_hand.jpg", img_hand[:,:,::-1])
                img_face_h, img_face_w = img_face.shape
                # 截取的脸的图像上的box
                l_tmp1 = l - l_new
                r_tmp1 = r - l_new
                t_tmp1 = t - t_new
                b_tmp1 = b - t_new
                if img_face_w >= norm_w or img_face_h > norm_h:
                    continue
                rand_x = random.randint(0, norm_w - img_face_w)
                rand_y = random.randint(0, norm_h - img_face_h)
                # 如果贴到目标图上，脸的座标变为
                hand_l = rand_x + l_tmp1
                hand_r = rand_x + r_tmp1
                hand_t = rand_y + t_tmp1
                hand_b = rand_y + b_tmp1
                # box的iou参考意义u不大
                # box_iou = bbox_utils.compute_boxIOU([hand_l, hand_t, hand_r, hand_b], box_dst)
                # 计算新图是否遮挡了之前存在的脸，如果被遮挡的面积大于等于0.1,则不可粘贴
                imgwh_box_iou = bbox_utils.compute_imgwh_boxIOU(
                    [rand_x, rand_y, rand_x + img_face_w, rand_y + img_face_h], box_dst_try)
                if imgwh_box_iou < 0.1:
                    try:
                        imgdst_try[rand_y:rand_y + img_face_h, rand_x:rand_x + img_face_w] = copy.deepcopy(img_face)
                    except:
                        continue
                    box_dst_try.append([hand_l, hand_t, hand_r, hand_b])
                    paste_face_num += 1
                    id_img += 1
                    id_box += 1
                    if paste_face_num == num_face_choice:
                        break

                # imgdst_tmp = copy.deepcopy(imgdst_try)
                # imgdst_tmp[rand_y:rand_y+img_hand_h, rand_x:rand_x+img_hand_w, : ] = img_hand
        if paste_face_num == num_face_choice:
            imgdst = copy.deepcopy(imgdst_try)
            box_dst = copy.deepcopy(box_dst_try)

        if len(box_dst) == 0:
            # print ("len_box_dst=0", list_img[0].shape, type(list_box[0]), list_box[0])
            return list_img[0], np.array([list_box[0]]).astype(np.float32)
        else:
            # imgdst_tmp = copy.deepcopy(imgdst)
            # print ("imgdst_boxdst", imgdst_tmp.shape, box_dst)
            # print ("boxdst_len", len(box_dst))
            # for box_tmp in box_dst:
            #     l, t, r, b = map(int, box_tmp)
            #     cv2.rectangle(imgdst_tmp, (l, t), (r, b), (0, 0, 255), 5, 8, 0)
            # cv2.imwrite("/media/baidu/ssd2/ppyolo/6w_data/small_test/less_handNum6/testHand/" + str(i).zfill(5) +str(int(1000*time.time()))+ ".jpg", imgdst_tmp[:,:,::-1])
            return imgdst, np.array(box_dst).astype(np.float32)

    def apply_image(self, img1, img2, factor):
        """
        :param img1:
        :param img2:
        :param factor:
        :return:
        """
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        img = np.zeros((h, w, img1.shape[2]), 'float32')
        img[:img1.shape[0], :img1.shape[1], :] = \
            img1.astype('float32') * factor
        img[:img2.shape[0], :img2.shape[1], :] += \
            img2.astype('float32') * (1.0 - factor)
        return img.astype('uint8')

    def mergeNew(self, list_img, list_box):
        '''
        利用方框截取标签，并拼接成新图像
        '''
        # img_merge = None
        # ggbox_merge = None
        box_list = []
        img0, img1, img2 = list_img
        # img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
        # img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("/home/baidu/Desktop/6faces/m0.jpg", img0)
        # cv2.imwrite("/home/baidu/Desktop/6faces/m1.jpg", img1)
        # cv2.imwrite("/home/baidu/Desktop/6faces/m2.jpg", img2)
        box0, box1, box2 = list_box
        box0 = list(box0[0])
        # print("list_box", box0)
        # print("list_box", box1)
        # print("list_box", box2)
        l0, t0, r0, b0 = map(int, box0)
        l1, t1, r1, b1 = map(int, box1)
        l2, t2, r2, b2 = map(int, box2)
        # cv2.rectangle(img0, (l0, t0), (r0, b0), (0, 0, 255), 5)
        # cv2.rectangle(img1, (l1, t1), (r1, b1), (0, 0, 255), 5)
        # cv2.rectangle(img2, (l2, t2), (r2, b2), (0, 0, 255), 5)
        # cv2.imwrite("/home/baidu/Desktop/6faces/model0.jpg", img0)
        # cv2.imwrite("/home/baidu/Desktop/6faces/model1.jpg", img1)
        # cv2.imwrite("/home/baidu/Desktop/6faces/model2.jpg", img2)
        imgdst = copy.deepcopy(img0)
        box_list.append([l0, t0, r0, b0])
        fscale = random.uniform(0, 1)

        # 取两张手的数据，每个手640×720,拼接成1280x720的图片
        # if fscale < 0.4:
        #     imgdst = np.ones((720, 1280, 3), dtype=np.uint8)
        #     h0,w0,_ = img0.shape
        #     h1,w1,_ = img1.shape
        #
        #     centx0 = (l0+r0)/2.
        #     centy0 = (t0+b0)/2.
        #     box_w0, box_h0 = r0-l0,t0-b0
        #     box_w0_new = 640
        #     box_h0_new = 720
        #
        #     centx1 = (l1 + r1) / 2.
        #     centy1 = (t1 + b1) / 2.
        #     box_w1, box_h1 = r1 - l1, t1 - b1
        #     box_w1_new = 640
        #     box_h1_new = 720
        #
        #     if box_w0_new > 640 or box_w1_new > 640:
        #         img_merge = img0
        #         ggbox_merge = box0
        #     else:
        #         x00_new = max(0, int(centx0 - box_w0_new//2))
        #         y00_new = max(0, int(centy0 - box_h0_new//2))
        #         x01_new = min(int(x00_new + 640), w0)
        #         y01_new = min(int(y00_new + 720), h0)
        #         # print ("new ltrb", x00_new, y00_new, x01_new, y01_new)
        #         img0_new = img0[y00_new:y01_new, x00_new:x01_new, :]
        #         l0_new, t0_new, r0_new, b0_new = l0 - x00_new, t0-y00_new, r0 - x00_new, b0-y00_new
        #         imgdst[:(y01_new-y00_new), :(x01_new-x00_new), :] = img0_new
        #         # cv2.rectangle(imgdst, (l0_new, t0_new), (r0_new, b0_new), (0,0,255), 5, 8, 0)
        #
        #         x10_new = max(0, int(centx1 - box_w1_new // 2))
        #         y10_new = max(0, int(centy1 - box_h1_new // 2))
        #         x11_new = min(int(x10_new + 640), w1)
        #         y11_new = min(int(y10_new + 720), h1)
        #         # print("new ltrb", x10_new, y10_new, x11_new, y11_new)
        #         img1_new = img1[y10_new:y11_new, x10_new:x11_new, :]
        #         l1_new, t1_new, r1_new, b1_new = l1 - x10_new+640, t1-y10_new, r1 - x10_new+640, b1-y10_new
        #         imgdst[:(y11_new-y10_new), 640:(x11_new-x10_new+640), :] = img1_new
        #         # cv2.rectangle(imgdst, (l1_new, t1_new), (r1_new, b1_new), (0, 255, 0), 5, 8, 0)
        #         # cv2.imwrite("/home/baidu/Desktop/models/model"+str(int(time.time()*1000))+".jpg", imgdst)
        #         box_list.append([l0_new, t0_new, r0_new, b0_new])
        #         box_list.append([l1_new, t1_new, r1_new, b1_new])

        if fscale < 0.5:  # 两个脸：一个脸不变，另一个脸随机贴
            # imgdst = copy.deepcopy(img0)
            # box_list.append([l0, t0, r0, b0])
            h0, w0 = img0.shape
            h1, w1 = img1.shape
            h2, w2 = img1.shape

            centx1 = (l1 + r1) / 2.
            centy1 = (t1 + b1) / 2.
            box_w1, box_h1 = r1 - l1, b1 - t1

            bool_left, bool_right = False, False
            for try_id in range(20):  # 超过最大次数，则不添加第二个人脸了
                fratio1 = random.uniform(2, 6)
                box_w1_new, box_h1_new = box_w1 * fratio1, box_h1 * fratio1
                l10 = max(0, int(centx1 - box_w1_new / 2.))
                t10 = max(0, int(centy1 - box_h1_new / 2.))
                r10 = min(int(centx1 + box_w1_new / 2.), w1)
                b10 = min(int(centy1 + box_h1_new / 2.), h1)
                box_w1_new, box_h1_new = r10 - l10, b10 - t10
                # print ("box new", l10, t10, r10, b10)
                if random.uniform(0, 1) < 0.5:  # 判断左边能否放下第二个人脸
                    if box_w1_new < l0 and box_h1_new < h0:
                        l_start = random.randint(0, l0 - box_w1_new)
                        t_start = random.randint(0, h0 - box_h1_new)
                        try:
                            imgdst[t_start:t_start + box_h1_new, l_start:l_start + box_w1_new] = img1[t10:b10,
                                                                                                    l10:r10]
                        except:
                            continue
                        l1_new = l1 - l10 + l_start
                        t1_new = t1 - t10 + t_start
                        r1_new = r1 - l10 + l_start
                        b1_new = b1 - t10 + t_start
                        bool_left = True
                        # cv2.rectangle(imgdst, (l1_new, t1_new), (r1_new, b1_new), (0, 255, 0), 5, 8, 0)

                else:  # 判断右边能否放下第二个人脸
                    if w0 - r0 > box_w1_new and box_h1_new < h0:
                        l_start = random.randint(r0, w0 - box_w1_new)
                        t_start = random.randint(0, h0 - box_h1_new)
                        try:
                            imgdst[t_start:t_start + box_h1_new, l_start:l_start + box_w1_new] = img1[t10:b10,
                                                                                                    l10:r10]
                        except:
                            continue
                        l1_new = l1 - l10 + l_start
                        t1_new = t1 - t10 + t_start
                        r1_new = r1 - l10 + l_start
                        b1_new = b1 - t10 + t_start
                        bool_right = True
                        # cv2.rectangle(imgdst, (l1_new, t1_new), (r1_new, b1_new), (0, 255, 0), 5, 8, 0)

                if bool_left or bool_right:
                    # cv2.rectangle(imgdst, (l0, t0), (r0, b0), (0, 0, 255), 5, 8, 0)
                    # cv2.imwrite("/home/baidu/Desktop/models/model" + str(int(time.time() * 1000)) + ".jpg", imgdst)
                    box_list.append([l1_new, t1_new, r1_new, b1_new])
                    break

        else:  # 三个人脸：一个人脸不变，另两个人脸随机贴
            # imgdst = copy.deepcopy(img0)
            # cv2.rectangle(imgdst, (l0, t0), (r0, b0), (0,255,255), 3, 8, 0)
            # box_list.append([l0, t0, r0, b0])
            h0, w0 = img0.shape
            h1, w1 = img1.shape
            h2, w2 = img2.shape

            bool_left, bool_right = False, False
            for try_id in range(30):  # 超过最大次数，则不添加第二，三个人脸了
                if not bool_left:
                    centx1 = (l1 + r1) / 2.
                    centy1 = (t1 + b1) / 2.
                    box_w1, box_h1 = r1 - l1, b1 - t1
                    fratio1 = random.uniform(2, 6)
                    box_w1_new, box_h1_new = box_w1 * fratio1, box_h1 * fratio1
                    l10 = max(0, int(centx1 - box_w1_new / 2.))
                    t10 = max(0, int(centy1 - box_h1_new / 2.))
                    r10 = min(int(centx1 + box_w1_new / 2.), w1)
                    b10 = min(int(centy1 + box_h1_new / 2.), h1)
                    box_w1_new, box_h1_new = r10 - l10, b10 - t10
                    # print ("box new", l10, t10, r10, b10)
                    # 判断左边能否放下第二个人脸
                    if box_w1_new < l0 and box_h1_new < h0:
                        l_start = random.randint(0, l0 - box_w1_new)
                        t_start = random.randint(0, h0 - box_h1_new)
                        try:
                            imgdst[t_start:t_start + box_h1_new, l_start:l_start + box_w1_new] = img1[t10:b10,
                                                                                                    l10:r10]
                        except:
                            continue
                        l1_new = l1 - l10 + l_start
                        t1_new = t1 - t10 + t_start
                        r1_new = r1 - l10 + l_start
                        b1_new = b1 - t10 + t_start
                        bool_left = True
                        # cv2.rectangle(imgdst, (l1_new, t1_new), (r1_new, b1_new), (0, 255, 0), 5, 8, 0)
                        box_list.append([l1_new, t1_new, r1_new, b1_new])
                if not bool_right:
                    centx2 = (l2 + r2) / 2.
                    centy2 = (t2 + b2) / 2.
                    box_w2, box_h2 = r2 - l2, b2 - t2
                    fratio2 = random.uniform(2, 6)
                    box_w2_new, box_h2_new = box_w2 * fratio2, box_h2 * fratio2
                    l20 = max(0, int(centx2 - box_w2_new / 2.))
                    t20 = max(0, int(centy2 - box_h2_new / 2.))
                    r20 = min(int(centx2 + box_w2_new / 2.), w2)
                    b20 = min(int(centy2 + box_h2_new / 2.), h2)
                    box_w2_new, box_h2_new = r20 - l20, b20 - t20
                    # 判断右边能否放下第三个人脸
                    if w0 - r0 > box_w2_new and box_h2_new < h0:
                        l_start = random.randint(r0, w0 - box_w2_new)
                        t_start = random.randint(0, h0 - box_h2_new)
                        try:
                            imgdst[t_start:t_start + box_h2_new, l_start:l_start + box_w2_new] = img2[t20:b20,
                                                                                                    l20:r20]
                        except:
                            continue
                        l2_new = l2 - l20 + l_start
                        t2_new = t2 - t20 + t_start
                        r2_new = r2 - l20 + l_start
                        b2_new = b2 - t20 + t_start
                        bool_right = True
                        # cv2.rectangle(imgdst, (l2_new, t2_new), (r2_new, b2_new), (255, 0, 0), 5, 8, 0)
                        box_list.append([l2_new, t2_new, r2_new, b2_new])

                # if bool_left or bool_right:
                #     cv2.imwrite("/home/baidu/Desktop/models/model" + str(int(time.time() * 1000)) + ".jpg", imgdst)

                if bool_left and bool_right:
                    # cv2.rectangle(imgdst, (l0, t0), (r0, b0), (0, 0, 255), 5, 8, 0)
                    # cv2.imwrite("/home/baidu/Desktop/models/model" + str(int(time.time() * 1000)) + ".jpg", imgdst)
                    break

        # 要转为float32, numpy
        # np.clip(box, 0., 1.)
        # img_merge = cv2.cvtColor(img_merge, cv2.COLOR_BGR2RGB)
        img_merge = copy.deepcopy(imgdst)
        ggbox_merge = np.array(box_list).astype(np.float32)

        # ggbox_merge[ggbox_merge[:, 0] < 0] = 0.
        # ggbox_merge[ggbox_merge[:, 0] > imgdst.shape[1]] = imgdst.shape[1]
        # ggbox_merge[ggbox_merge[:, 2] < 0] = 0.
        # ggbox_merge[ggbox_merge[:, 2] > imgdst.shape[1]] = imgdst.shape[1]
        #
        # ggbox_merge[ggbox_merge[:, 1] < 0] = 0.
        # ggbox_merge[ggbox_merge[:, 1] > imgdst.shape[0]] = imgdst.shape[0]
        # ggbox_merge[ggbox_merge[:, 3] < 0] = 0.
        # ggbox_merge[ggbox_merge[:, 3] > imgdst.shape[0]] = imgdst.shape[0]
        # ggbox_merge = np.array(box_list).astype(np.float32)
        # ggbox_merge = np.clip(ggbox_merge, 0.)Mixup_ImgnetMixup_Imgnet

        return img_merge, ggbox_merge

    def __call__(self, sample, context=None):
        result = copy.deepcopy(sample)
        # print ("keys", sample.keys())
        # result['only_negimg'] = 0
        # print ("*"*100)
        # print (result)
        # print ("sample", sample)
        if sample['gt_bbox'].shape[0] > 1 or sample['gt_bbox'].shape[0] == 0:
            return result
        f_ratio = random.uniform(0, 1)
        if f_ratio < 0.2:
            return result
        elif f_ratio < 0.6:
            allNums = len(self.allPathInfos)
            id1 = random.randint(0, allNums - 1)
            id2 = random.randint(0, allNums - 1)
            img1, box1 = self.get_img_box_points(self.allPathInfos[id1])
            img2, box2 = self.get_img_box_points(self.allPathInfos[id2])
            img0, box0 = sample['image'], sample['gt_bbox']
            if len(box1) == 0 or len(box2) == 0:
                # print ("sample:", sample)
                return sample

            # img0 = cv.cvtColor(img0, cv.COLOR_RGB2BGR)
            # img1 = cv.cvtColor(img1, cv.COLOR_RGB2BGR)
            # img2 = cv.cvtColor(img2, cv.COLOR_RGB2BGR)
            # cv2.imwrite("/home/baidu/Desktop/6faces/n_0_" + str(int(1000 * (time.time()))) + ".jpg", img0)
            # cv2.imwrite("/home/baidu/Desktop/6faces/n_1_" + str(int(1000 * (time.time()))) + ".jpg", img1)
            # cv2.imwrite("/home/baidu/Desktop/6faces/n_2_" + str(int(1000 * (time.time()))) + ".jpg", img2)

            newImage, newBox = self.mergeNew([img0, img1, img2], [box0, box1, box2])
            result['image'] = newImage
            result["gt_bbox"] = newBox
            result["gt_score"] = np.ones(shape=(newBox.shape[0], 1), dtype=np.float32)
            result["gt_class"] = np.zeros(shape=(newBox.shape[0], 1), dtype=np.float32)
            # print ("newBox:",newBox.shape)
            # for i in range(newBox.shape[0]):
            #     l,t,r,b = newBox[i]
            #     l,t,r,b = map(int, [l,t,r,b])
            #     cv.rectangle(newImage, (l,t), (r,b), (255,0,0), 5, 8, 0)
            # cv2.imwrite("/home/baidu/Desktop/6faces/new_"+str(int(1000*(time.time())) )+".jpg", newImage[:,:,::-1])
            # print ("result:", result)
            return result
        elif f_ratio < 0.8:
            allNums = len(self.allPathInfos)
            img0, box0 = sample['image'], sample['gt_bbox']
            imgsTmp1 = []
            boxTmp1 = []
            imgsTmp1.append(img0)
            boxTmp1.append(box0)
            for i in range(4):
                id_item = random.randint(0, allNums - 1)
                img_item, box_item = self.get_img_box_points(self.allPathInfos[id_item])
                if len(box_item) == 0:
                    return sample
                imgsTmp1.append(img_item)
                boxTmp1.append(box_item)

            # newImage, newBbox = operators_MergeNew.mergeImages(imgsTmp1, boxTmp1, self.negFiles)
            newImage, newBbox = self.mergeNew_5Faces(imgsTmp1, boxTmp1, self.negFiles)
            result['image'] = newImage
            result["gt_bbox"] = newBbox
            result["gt_score"] = np.ones(shape=(newBbox.shape[0], 1), dtype=np.float32)
            result["gt_class"] = np.zeros(shape=(newBbox.shape[0], 1), dtype=np.float32)
            # for id in range(newBbox.shape[0]):
            #     l,t,r,b = newBbox[id]
            #     l,t,r,b = map(int, [l,t,r,b])
            #     cv2.rectangle(newImage, (l,t), (r,b), (0,255,255), 3, 8, 0)
            # cv2.imwrite("/home/baidu/Desktop/6faces/"+str(int(1000*(time.time())))+".jpg", newImage[:,:,::-1])
            return result
        else:  # 纯背景图
            frandom = random.uniform(0, 1)
            if frandom < 0.5:  # 选取负样本作为纯背景图
                result1 = copy.deepcopy(result)
                # print ("negFiles")
                nid = random.randint(0, len(self.negFiles) - 1)
                negImg = cv2.imread(self.negFiles[nid], 0)
                h = result['h']
                w = result['w']
                negImg3 = cv2.resize(negImg, (w, h))
                # negImg3 = cv2.cvtColor(negImg2, cv2.COLOR_BGR2RGB)
                # for item in result.keys():
                #     if item in ['gt_bbox', 'gt_score', 'gt_class']:
                #         continue
                #     result1[item] = result[item]
                result1['image'] = negImg3
                # result1['gt_bbox'] = None
                # result1['gt_score'] = None
                # result1['gt_class'] = None
                result1['only_negimg'] = 1
            else:  # 正样本遮挡人脸作为负样本
                result1 = copy.deepcopy(result)
                # print ("negFiles")
                # nid =random.randint(0, len(self.negFiles)-1)
                # negImg = cv2.imread(self.negFiles[nid])

                try:
                    img_t1 = result1['image']
                    box_t1 = result1['gt_bbox']
                    l, t, r, b = list(box_t1[0])
                    l, t, r, b = map(int, [l, t, r, b])
                    if random.uniform(0, 1) < 0.5:
                        int_r = random.randint(0, 255)
                        # int_g = random.randint(0, 255)
                        # int_b = random.randint(0, 255)
                    else:
                        int_r = random.randint(0, 255)
                        # int_g = int_r
                        # int_b = int_r
                    imgmask = np.ones((b - t, r - l), dtype=np.uint8)
                    imgmask[:, :] = int_r
                    # imgmask[:, :, 1] = int_g
                    # imgmask[:, :, 2] = int_b
                    img_t1[t:b, l:r] = imgmask

                    # h = result['h']
                    # w = result['w']
                    # negImg2 = cv2.resize(negImg, (w, h))
                    # negImg3 = cv2.cvtColor(negImg2, cv2.COLOR_BGR2RGB)
                    # for item in result.keys():
                    #     if item in ['gt_bbox', 'gt_score', 'gt_class']:
                    #         continue
                    #     result1[item] = result[item]
                    result1['image'] = copy.deepcopy(img_t1)
                    # result1['gt_bbox'] = None
                    # result1['gt_score'] = None
                    # result1['gt_class'] = None
                    result1['only_negimg'] = 1
                except:
                    result['only_negimg'] = 0
                    return result
            # cv.imwrite("/home/baidu/Desktop/6faces/neg_" + str(int(100*(time.time()))) + ".jpg", result1['image'])
            return result1


class RandomRotate(BaseOperator):
    """
    Mixup image and gt_bbbox/gt_score
    """

    def __init__(self, prob=0.5):
        """ Mixup_Imgnet image and gt_score
        """
        super(RandomRotate, self).__init__()
        self.prob = prob

    def __call__(self, sample, context=None):
        if isinstance(sample, Sequence):
            sample = sample[0]
        if random.uniform(0, 1) < self.prob:
            return sample
        if 'gt_bbox' not in sample:
            return sample
        if sample['gt_bbox'] is None or len(sample['gt_bbox']) == 0:
            return sample

        # print ("Mixup_Imgnet", sample)

        imgori = sample['image']
        BoxOri = sample['gt_bbox']
        # imgori1 = cv2.cvtColor(imgori, cv2.COLOR_RGB2BGR)
        imgori1 = copy.deepcopy(imgori)
        # print ("BoxOri", BoxOri)
        # for i in range(BoxOri.shape[0]):
        #     box = BoxOri[i]
        #     l,t,r,b = map(int, box)
        #     cv2.rectangle(imgori1, (l,t), (r,b), (0,0,255), 5, 8, 0)
        # cv2.imwrite("/home/baidu/Desktop/models/rot_" + str(int(1000 * (time.time()))) +"_shape0_" + str(BoxOri.shape[0])+ ".jpg", imgori1)

        h, w = imgori1.shape
        centx, centy = w / 2., h / 2.
        if BoxOri.shape[0] == 1 and random.uniform(0, 1) < 0.5:
                fAngle = random.uniform(-60., 60.)
        else:
            fAngle = random.uniform(-20., 20.)

        rotMat = cv2.getRotationMatrix2D((centx, centy), fAngle, 1.)
        imgRotate1 = cv2.warpAffine(imgori1, rotMat, (w, h))
        pt1_ori = BoxOri[:, :2]  # l,t
        pt1_ori2 = np.transpose(pt1_ori, (1, 0))
        pt1_ori2_tmp = np.ones(shape=(3, pt1_ori2.shape[1]))
        pt1_ori2_tmp[:2, :] = pt1_ori2
        pt1_rot1 = np.dot(rotMat, pt1_ori2_tmp)
        pt1_rot2 = np.transpose(pt1_rot1, (1, 0))
        # print ("pt1_rot2:", pt1_rot2.shape,  pt1_rot2)
        # for i in range(pt1_rot2.shape[0]):
        #     x,y = int(pt1_rot2[i][0]),int(pt1_rot2[i][1])
        #     cv2.circle(imgRotate1, (x,y), 15, (0,255,255), -1, 8, 0)

        pt2_ori = BoxOri[:, 2:]  # r,b
        pt2_ori2 = np.transpose(pt2_ori, (1, 0))
        pt2_ori2_tmp = np.ones(shape=(3, pt2_ori2.shape[1]))
        pt2_ori2_tmp[:2, :] = pt2_ori2
        pt2_rot1 = np.dot(rotMat, pt2_ori2_tmp)
        pt2_rot2 = np.transpose(pt2_rot1, (1, 0))
        # print("pt2_rot2:", pt2_rot2.shape, pt2_rot2)
        # for i in range(pt2_rot2.shape[0]):
        #     x, y = int(pt2_rot2[i][0]), int(pt2_rot2[i][1])
        #     cv2.circle(imgRotate1, (x, y), 15, (0, 255, 255), -1, 8, 0)

        pt3_ori = BoxOri[:, [2, 1]]  # r,t
        pt3_ori2 = np.transpose(pt3_ori, (1, 0))
        pt3_ori2_tmp = np.ones(shape=(3, pt3_ori2.shape[1]))
        pt3_ori2_tmp[:2, :] = pt3_ori2
        pt3_rot1 = np.dot(rotMat, pt3_ori2_tmp)
        pt3_rot2 = np.transpose(pt3_rot1, (1, 0))

        pt4_ori = BoxOri[:, [0, 3]]  # l,b
        pt4_ori2 = np.transpose(pt4_ori, (1, 0))
        pt4_ori2_tmp = np.ones(shape=(3, pt4_ori2.shape[1]))
        pt4_ori2_tmp[:2, :] = pt4_ori2
        pt4_rot1 = np.dot(rotMat, pt4_ori2_tmp)
        pt4_rot2 = np.transpose(pt4_rot1, (1, 0))

        box_rot = copy.deepcopy(BoxOri)
        # print ("box_rot", box_rot.shape, pt1_rot2.shape, pt2_rot2.shape)
        # box_rot[:, :2] = pt1_rot2
        # box_rot[:, 2:] = pt2_rot2
        for i in range(box_rot.shape[0]):
            box_rot[i][0] = min(pt1_rot2[i][0], pt2_rot2[i][0], pt3_rot2[i][0], pt4_rot2[i][0])
            box_rot[i][1] = min(pt1_rot2[i][1], pt2_rot2[i][1], pt3_rot2[i][1], pt4_rot2[i][1])
            box_rot[i][2] = max(pt1_rot2[i][0], pt2_rot2[i][0], pt3_rot2[i][0], pt4_rot2[i][0])
            box_rot[i][3] = max(pt1_rot2[i][1], pt2_rot2[i][1], pt3_rot2[i][1], pt4_rot2[i][1])

        # for id in range(box_rot1.shape[0]):
        #     item = box_rot1[id]
        #     l,t,r,b = map(int, item.tolist())
        #     cv2.rectangle(imgRotate1, (l,t), (r,b), (0,0,255), 3, 8, 0)

        # box_rot[box_rot[:, 0] < 0] = 0.
        # box_rot[box_rot[:, 0] > imgori.shape[1]] = imgori.shape[1]
        # box_rot[box_rot[:, 2] < 0] = 0.
        # box_rot[box_rot[:, 2] > imgori.shape[1]] = imgori.shape[1]
        # box_rot[box_rot[:, 1] < 0] = 0.
        # box_rot[box_rot[:, 1] > imgori.shape[0]] = imgori.shape[0]
        # box_rot[box_rot[:, 3] < 0] = 0.
        # box_rot[box_rot[:, 3] > imgori.shape[0]] = imgori.shape[0]

        # for i in range(box_rot.shape[0]):
        #     print ("*"*100)
        #     box = box_rot[i]
        #     print (box)
        #     l,t,r,b = map(int, box)
        #     cv2.rectangle(imgRotate1, (l,t), (r,b), (0,0,255), 5, 8, 0)
        # cv2.imwrite("/home/baidu/Desktop/t1/rot_" + str(int(1000 * (time.time()))) +"_shape0_" + str(BoxOri.shape[0])+ ".jpg", imgRotate1)

        box_rot = np.array(box_rot).astype(np.float32)
        result = copy.deepcopy(sample)
        result["image"] = imgRotate1[:, :]
        result["gt_bbox"] = box_rot

        return result


class RandomBlurNoise(BaseOperator):
    """
    Mixup image and gt_bbbox/gt_score
    """

    def __init__(self, prob=0.5):
        """ Mixup_Imgnet image and gt_score
        """
        super(RandomBlurNoise, self).__init__()
        self.prob = prob

    def __call__(self, sample, context=None):
        if isinstance(sample, Sequence):
            sample = sample[0]

        # if random.uniform(0, 1) < self.prob:
        #     return sample
        result = copy.deepcopy(sample)
        # print ("Mixup_Imgnet", sample)
        imgori = sample['image']
        # imgori1 = cv2.cvtColor(imgori, cv2.COLOR_RGB2BGR)
        imgori2 = copy.deepcopy(imgori)
        if random.uniform(0, 1) < self.prob:
            imgori2 = cv2.blur(imgori2, (3, 3))
            # imgori2 = cv2.cvtColor(imgori2, cv2.COLOR_BGR2RGB)
        if random.uniform(0, 1) < self.prob:
            # if random.uniform(0,1)<0.5:
            #     ""
            # else:
            #     ""
            # gaosi  noise
            noise = np.random.normal(0, 0.001, imgori.shape)
            imgtmp1 = np.array(imgori).astype(np.float32)
            imgtmp2 = imgtmp1 + noise
            imgtmp2[imgtmp2 > 255] = 255
            imgtmp2[imgtmp2 < 0] = 0
            imgori2 = np.array(imgtmp2).astype(np.uint8)

        # cv2.imwrite("/home/baidu/Desktop/models/mixBack.jpg", imMix[:,:,::-1])

        result["image"] = imgori2
        return result


class Mixup_Imgnet(BaseOperator):
    """
    Mixup image and gt_bbbox/gt_score
    """

    def __init__(self, back_imgs=[], prob=0.1):
        """ Mixup_Imgnet image and gt_score
        """
        super(Mixup_Imgnet, self).__init__()
        self.backImgs = back_imgs
        self.backNum = len(self.backImgs)
        self.prob = prob

    def __call__(self, sample, context=None):
        if random.uniform(0, 1) < self.prob:
            return sample
        imgori = sample['image']
        factor = random.uniform(0.8, 1)
        randId = random.randint(0, self.backNum - 1)
        imgback = cv2.imread(self.backImgs[randId], 0)
        imgback = cv2.resize(imgback, (imgori.shape[1], imgori.shape[0]))
        # imgori1 = cv2.cvtColor(imgori, cv2.COLOR_RGB2BGR)
        imgori1 = copy.deepcopy(imgori)

        # if random.uniform(0, 1) < 0.5:
        #     imgback = cv.cvtColor(imgback, cv.COLOR_BGR2GRAY)
        #     imgback = cv.cvtColor(imgback, cv.COLOR_GRAY2BGR)
        imMix = cv2.addWeighted(imgori1, factor, imgback, 1 - factor, 0.)

        # bbox = sample['gt_bbox']
        # for i in range(bbox.shape[0]):
        #     box = bbox[i]
        #     l, t,r,b = map(int, box)
        #     cv2.rectangle(imMix1, (l,t), (r,b), (0,0,255), 10, 8, 0)
        # cv2.imwrite("/home/baidu/Desktop/models/mix_" + str(1000*(time.time())) + ".jpg", imMix1)

        # imMix = cv2.cvtColor(imMix1, cv2.COLOR_BGR2RGB)
        # im = self.apply_image(sample[0]['image'], sample[1]['image'], factor)
        result = copy.deepcopy(sample)
        result['image'] = imMix
        # apply bbox and score

        if 'gt_score' in sample:
            if sample['gt_score'] is None:
                ""
            else:
                result['gt_score'] *= factor
        # cv2.imwrite("/home/baidu/Desktop/models/mixBack.jpg", imMix[:,:,::-1])

        return result


# @register_op
class NormalizeBox(BaseOperator):
    """Transform the bounding box's coornidates to [0,1]."""

    def __init__(self):
        super(NormalizeBox, self).__init__()

    def apply(self, sample, context):
        if 'gt_bbox' not in sample:
            return sample
        if sample['gt_bbox'] is None or len(sample['gt_bbox']) == 0:
            return sample

        im = sample['image']
        gt_bbox = sample['gt_bbox']
        height, width = im.shape
        for i in range(gt_bbox.shape[0]):
            gt_bbox[i][0] = gt_bbox[i][0] / width
            gt_bbox[i][1] = gt_bbox[i][1] / height
            gt_bbox[i][2] = gt_bbox[i][2] / width
            gt_bbox[i][3] = gt_bbox[i][3] / height
        sample['gt_bbox'] = gt_bbox

        if 'gt_keypoint' in sample.keys():
            gt_keypoint = sample['gt_keypoint']

            for i in range(gt_keypoint.shape[1]):
                if i % 2:
                    gt_keypoint[:, i] = gt_keypoint[:, i] / height
                else:
                    gt_keypoint[:, i] = gt_keypoint[:, i] / width
            sample['gt_keypoint'] = gt_keypoint

        return sample


# @register_op
class BboxXYXY2XYWH(BaseOperator):
    """
    Convert bbox XYXY format to XYWH format.
    """

    def __init__(self):
        super(BboxXYXY2XYWH, self).__init__()

    def apply(self, sample, context=None):
        if 'gt_bbox' not in sample:
            return sample
        if sample['gt_bbox'] is None or len(sample['gt_bbox']) == 0:
            return sample
        assert 'gt_bbox' in sample
        bbox = sample['gt_bbox']
        bbox[:, 2:4] = bbox[:, 2:4] - bbox[:, :2]
        bbox[:, :2] = bbox[:, :2] + bbox[:, 2:4] / 2.
        sample['gt_bbox'] = bbox
        return sample


# @register_op
class PadBox(BaseOperator):
    """
    Pad zeros to bboxes if number of bboxes is less than num_max_boxes.
    """

    def __init__(self, num_max_boxes=50):
        """
        Pad zeros to bboxes if number of bboxes is less than num_max_boxes.
        Args:
            num_max_boxes (int): the max number of bboxes
        """
        self.num_max_boxes = num_max_boxes
        super(PadBox, self).__init__()

    def apply(self, sample, context=None):
        if 'gt_bbox' not in sample:
            return sample
        # if sample['gt_bbox'] is None or len(sample['gt_bbox']) == 0:
        if sample['gt_bbox'] is None:
            return sample

        assert 'gt_bbox' in sample
        bbox = sample['gt_bbox']
        gt_num = min(self.num_max_boxes, len(bbox))
        num_max = self.num_max_boxes
        # fields = context['fields'] if context else []
        pad_bbox = np.zeros((num_max, 4), dtype=np.float32)
        if gt_num > 0:
            pad_bbox[:gt_num, :] = bbox[:gt_num, :]
        sample['gt_bbox'] = pad_bbox
        if 'gt_class' in sample:
            pad_class = np.zeros((num_max,), dtype=np.int32)
            if gt_num > 0:
                pad_class[:gt_num] = sample['gt_class'][:gt_num, 0]
            sample['gt_class'] = pad_class
        if 'gt_score' in sample:
            pad_score = np.zeros((num_max,), dtype=np.float32)
            if gt_num > 0:
                pad_score[:gt_num] = sample['gt_score'][:gt_num, 0]
            sample['gt_score'] = pad_score
        # in training, for example in op ExpandImage,
        # the bbox and gt_class is expandded, but the difficult is not,
        # so, judging by it's length
        if 'difficult' in sample:
            pad_diff = np.zeros((num_max,), dtype=np.int32)
            if gt_num > 0:
                pad_diff[:gt_num] = sample['difficult'][:gt_num, 0]
            sample['difficult'] = pad_diff
        if 'is_crowd' in sample:
            pad_crowd = np.zeros((num_max,), dtype=np.int32)
            if gt_num > 0:
                pad_crowd[:gt_num] = sample['is_crowd'][:gt_num, 0]
            sample['is_crowd'] = pad_crowd
        return sample


# @register_op
class DebugVisibleImage(BaseOperator):
    """
    In debug mode, visualize images according to `gt_box`.
    (Currently only supported when not cropping and flipping image.)
    """

    def __init__(self, output_dir='output/debug', is_normalized=False):
        super(DebugVisibleImage, self).__init__()
        self.is_normalized = is_normalized
        self.output_dir = output_dir
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        if not isinstance(self.is_normalized, bool):
            raise TypeError("{}: input type is invalid.".format(self))

    def apply(self, sample, context=None):
        image = Image.open(sample['im_file']).convert('RGB')
        out_file_name = sample['im_file'].split('/')[-1]
        width = sample['w']
        height = sample['h']
        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']
        draw = ImageDraw.Draw(image)
        for i in range(gt_bbox.shape[0]):
            if self.is_normalized:
                gt_bbox[i][0] = gt_bbox[i][0] * width
                gt_bbox[i][1] = gt_bbox[i][1] * height
                gt_bbox[i][2] = gt_bbox[i][2] * width
                gt_bbox[i][3] = gt_bbox[i][3] * height

            xmin, ymin, xmax, ymax = gt_bbox[i]
            draw.line(
                [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
                 (xmin, ymin)],
                width=2,
                fill='green')
            # draw label
            text = str(gt_class[i][0])
            tw, th = draw.textsize(text)
            draw.rectangle(
                [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill='green')
            draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))

        if 'gt_keypoint' in sample.keys():
            gt_keypoint = sample['gt_keypoint']
            if self.is_normalized:
                for i in range(gt_keypoint.shape[1]):
                    if i % 2:
                        gt_keypoint[:, i] = gt_keypoint[:, i] * height
                    else:
                        gt_keypoint[:, i] = gt_keypoint[:, i] * width
            for i in range(gt_keypoint.shape[0]):
                keypoint = gt_keypoint[i]
                for j in range(int(keypoint.shape[0] / 2)):
                    x1 = round(keypoint[2 * j]).astype(np.int32)
                    y1 = round(keypoint[2 * j + 1]).astype(np.int32)
                    draw.ellipse(
                        (x1, y1, x1 + 5, y1 + 5), fill='green', outline='green')
        save_path = os.path.join(self.output_dir, out_file_name)
        image.save(save_path, quality=95)
        return sample


# @register_op
class Pad(BaseOperator):
    """
    Pad image to a specified size or multiple of size_divisor.
    """

    def __init__(self,
                 size=None,
                 size_divisor=32,
                 pad_mode=0,
                 offsets=None,
                 # fill_value=(127.5, 127.5, 127.5),
                fill_value = 127.5
                 ):
        """
        Pad image to a specified size or multiple of size_divisor.
        Args:
            size (int, Sequence): image target size, if None, pad to multiple of size_divisor, default None
            size_divisor (int): size divisor, default 32
            pad_mode (int): pad mode, currently only supports four modes [-1, 0, 1, 2]. if -1, use specified offsets
                if 0, only pad to right and bottom. if 1, pad according to center. if 2, only pad left and top
            offsets (list): [offset_x, offset_y], specify offset while padding, only supported pad_mode=-1
            fill_value (bool): rgb value of pad area, default (127.5, 127.5, 127.5)
        """
        super(Pad, self).__init__()

        if not isinstance(size, (int, Sequence)):
            raise TypeError(
                "Type of target_size is invalid when random_size is True. \
                            Must be List, now is {}".format(type(size)))

        if isinstance(size, int):
            size = [size, size]

        assert pad_mode in [
            -1, 0, 1, 2
        ], 'currently only supports four modes [-1, 0, 1, 2]'
        assert pad_mode == -1 and offsets, 'if pad_mode is -1, offsets should not be None'

        self.size = size
        self.size_divisor = size_divisor
        self.pad_mode = pad_mode
        self.fill_value = fill_value
        self.offsets = offsets

    def apply_segm(self, segms, offsets, im_size, size):
        """

        :param segms:
        :param offsets:
        :param im_size:
        :param size:
        :return:
        """

        def _expand_poly(poly, x, y):
            """

            :param poly:
            :param x:
            :param y:
            :return:
            """
            expanded_poly = np.array(poly)
            expanded_poly[0::2] += x
            expanded_poly[1::2] += y
            return expanded_poly.tolist()

        def _expand_rle(rle, x, y, height, width, h, w):
            """

            :param rle:
            :param x:
            :param y:
            :param height:
            :param width:
            :param h:
            :param w:
            :return:
            """
            if 'counts' in rle and type(rle['counts']) == list:
                rle = mask_util.frPyObjects(rle, height, width)
            mask = mask_util.decode(rle)
            expanded_mask = np.full((h, w), 0).astype(mask.dtype)
            expanded_mask[y:y + height, x:x + width] = mask
            rle = mask_util.encode(
                np.array(
                    expanded_mask, order='F', dtype=np.uint8))
            return rle

        x, y = offsets
        height, width = im_size
        h, w = size
        expanded_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                expanded_segms.append(
                    [_expand_poly(poly, x, y) for poly in segm])
            else:
                # RLE format
                import pycocotools.mask as mask_util
                expanded_segms.append(
                    _expand_rle(segm, x, y, height, width, h, w))
        return expanded_segms

    def apply_bbox(self, bbox, offsets):
        """

        :param bbox:
        :param offsets:
        :return:
        """
        return bbox + np.array(offsets * 2, dtype=np.float32)

    def apply_keypoint(self, keypoints, offsets):
        """

        :param keypoints:
        :param offsets:
        :return:
        """
        n = len(keypoints[0]) // 2
        return keypoints + np.array(offsets * n, dtype=np.float32)

    def apply_image(self, image, offsets, im_size, size):
        """

        :param image:
        :param offsets:
        :param im_size:
        :param size:
        :return:
        """
        x, y = offsets
        im_h, im_w = im_size
        h, w = size
        # canvas = np.ones((h, w, 3), dtype=np.float32)
        canvas = np.ones((h, w), dtype=np.float32)
        canvas *= np.array(self.fill_value, dtype=np.float32)
        canvas[y:y + im_h, x:x + im_w] = image.astype(np.float32)
        return canvas

    def apply(self, sample, context=None):
        im = sample['image']
        im_h, im_w = im.shape[:2]
        if self.size:
            h, w = self.size
            assert (
                    im_h < h and im_w < w
            ), '(h, w) of target size should be greater than (im_h, im_w)'
        else:
            h = np.ceil(im_h // self.size_divisor) * self.size_divisor
            w = np.ceil(im_w / self.size_divisor) * self.size_divisor

        if h == im_h and w == im_w:
            return sample

        if self.pad_mode == -1:
            offset_x, offset_y = self.offsets
        elif self.pad_mode == 0:
            offset_y, offset_x = 0, 0
        elif self.pad_mode == 1:
            offset_y, offset_x = (h - im_h) // 2, (w - im_w) // 2
        else:
            offset_y, offset_x = h - im_h, w - im_w

        offsets, im_size, size = [offset_x, offset_y], [im_h, im_w], [h, w]

        sample['image'] = self.apply_image(im, offsets, im_size, size)

        if self.pad_mode == 0:
            return sample
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], offsets)

        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(sample['gt_poly'], offsets,
                                                im_size, size)

        if 'gt_keypoint' in sample and len(sample['gt_keypoint']) > 0:
            sample['gt_keypoint'] = self.apply_keypoint(sample['gt_keypoint'],
                                                        offsets)

        return sample


# @register_op
class Poly2Mask(BaseOperator):
    """
    gt poly to mask annotations
    """

    def __init__(self):
        super(Poly2Mask, self).__init__()
        import pycocotools.mask as maskUtils
        self.maskutils = maskUtils

    def _poly2mask(self, mask_ann, img_h, img_w):
        """

        :param mask_ann:
        :param img_h:
        :param img_w:
        :return:
        """
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = self.maskutils.frPyObjects(mask_ann, img_h, img_w)
            rle = self.maskutils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = self.maskutils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = self.maskutils.decode(rle)
        return mask

    def apply(self, sample, context=None):
        assert 'gt_poly' in sample
        im_h = sample['h']
        im_w = sample['w']
        masks = [
            self._poly2mask(gt_poly, im_h, im_w)
            for gt_poly in sample['gt_poly']
        ]
        sample['gt_segm'] = np.asarray(masks).astype(np.uint8)
        return sample


# @register_op
class Rbox2Poly(BaseOperator):
    """
    Convert rbbox format to poly format.
    """

    def __init__(self):
        super(Rbox2Poly, self).__init__()

    def apply(self, sample, context=None):
        assert 'gt_rbox' in sample
        assert sample['gt_rbox'].shape[1] == 5
        rrects = sample['gt_rbox']
        x_ctr = rrects[:, 0]
        y_ctr = rrects[:, 1]
        width = rrects[:, 2]
        height = rrects[:, 3]
        x1 = x_ctr - width / 2.0
        y1 = y_ctr - height / 2.0
        x2 = x_ctr + width / 2.0
        y2 = y_ctr + height / 2.0
        sample['gt_bbox'] = np.stack([x1, y1, x2, y2], axis=1)
        polys = bbox_utils.rbox2poly(rrects)
        sample['gt_rbox2poly'] = polys
        return sample


# #@register_op
class PadBatch(BaseOperator):
    """
    Pad a batch of samples so they can be divisible by a stride.
    The layout of each image should be 'CHW'.
    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
    """

    def __init__(self, pad_to_stride=0, pad_gt=False):
        super(PadBatch, self).__init__()
        self.pad_to_stride = pad_to_stride
        self.pad_gt = pad_gt

    def __call__(self, samples, context=None):
        """
        Args:
            samples (list): a batch of sample, each is dict.
        """
        coarsest_stride = self.pad_to_stride

        max_shape = np.array([data['image'].shape for data in samples]).max(
            axis=0)
        if coarsest_stride > 0:
            max_shape[1] = int(
                np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
            max_shape[2] = int(
                np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)

        padding_batch = []
        for data in samples:
            im = data['image']
            im_c, im_h, im_w = im.shape[:]
            padding_im = np.zeros(
                (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = im
            data['image'] = padding_im
            if 'semantic' in data and data['semantic'] is not None:
                semantic = data['semantic']
                padding_sem = np.zeros(
                    (1, max_shape[1], max_shape[2]), dtype=np.float32)
                padding_sem[:, :im_h, :im_w] = semantic
                data['semantic'] = padding_sem
            if 'gt_segm' in data and data['gt_segm'] is not None:
                gt_segm = data['gt_segm']
                padding_segm = np.zeros(
                    (gt_segm.shape[0], max_shape[1], max_shape[2]),
                    dtype=np.uint8)
                padding_segm[:, :im_h, :im_w] = gt_segm
                data['gt_segm'] = padding_segm

        if self.pad_gt:
            gt_num = []
            if 'gt_poly' in data and data['gt_poly'] is not None and len(data[
                                                                             'gt_poly']) > 0:
                pad_mask = True
            else:
                pad_mask = False

            if pad_mask:
                poly_num = []
                poly_part_num = []
                point_num = []
            for data in samples:
                gt_num.append(data['gt_bbox'].shape[0])
                if pad_mask:
                    poly_num.append(len(data['gt_poly']))
                    for poly in data['gt_poly']:
                        poly_part_num.append(int(len(poly)))
                        for p_p in poly:
                            point_num.append(int(len(p_p) / 2))
            gt_num_max = max(gt_num)

            for i, data in enumerate(samples):
                gt_box_data = -np.ones([gt_num_max, 4], dtype=np.float32)
                gt_class_data = -np.ones([gt_num_max], dtype=np.int32)
                is_crowd_data = np.ones([gt_num_max], dtype=np.int32)
                difficult_data = np.ones([gt_num_max], dtype=np.int32)

                if pad_mask:
                    poly_num_max = max(poly_num)
                    poly_part_num_max = max(poly_part_num)
                    point_num_max = max(point_num)
                    gt_masks_data = -np.ones(
                        [poly_num_max, poly_part_num_max, point_num_max, 2],
                        dtype=np.float32)

                gt_num = data['gt_bbox'].shape[0]
                gt_box_data[0:gt_num, :] = data['gt_bbox']
                gt_class_data[0:gt_num] = np.squeeze(data['gt_class'])
                if 'is_crowd' in data:
                    is_crowd_data[0:gt_num] = np.squeeze(data['is_crowd'])
                    data['is_crowd'] = is_crowd_data
                if 'difficult' in data:
                    difficult_data[0:gt_num] = np.squeeze(data['difficult'])
                    data['difficult'] = difficult_data
                if pad_mask:
                    for j, poly in enumerate(data['gt_poly']):
                        for k, p_p in enumerate(poly):
                            pp_np = np.array(p_p).reshape(-1, 2)
                            gt_masks_data[j, k, :pp_np.shape[0], :] = pp_np
                    data['gt_poly'] = gt_masks_data
                data['gt_bbox'] = gt_box_data
                data['gt_class'] = gt_class_data

        return samples


# #@register_op
class BatchRandomResize(BaseOperator):
    """
    Resize image to target size randomly. random target_size and interpolation method
    Args:
        target_size (int, list, tuple): image target size, if random size is True, must be list or tuple
        keep_ratio (bool): whether keep_raio or not, default true
        interp (int): the interpolation method
        random_size (bool): whether random select target size of image
        random_interp (bool): whether random select interpolation method
    """

    def __init__(self,
                 target_size,
                 keep_ratio,
                 interp=cv2.INTER_NEAREST,
                 random_size=True,
                 random_interp=False):
        super(BatchRandomResize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]
        self.interp = interp
        assert isinstance(target_size, (
            int, Sequence)), "target_size must be int, list or tuple"
        if random_size and not isinstance(target_size, list):
            raise TypeError(
                "Type of target_size is invalid when random_size is True. Must be List, now is {}".
                    format(type(target_size)))
        self.target_size = target_size
        self.random_size = random_size
        self.random_interp = random_interp

    def __call__(self, samples, context=None):
        if self.random_size:
            target_size = random.choice(self.target_size)
        else:
            target_size = self.target_size

        if self.random_interp:
            interp = np.random.choice(self.interps)
        else:
            interp = self.interp

        resizer = Resize(target_size, keep_ratio=self.keep_ratio, interp=interp)
        return resizer(samples, context=context)


# #@register_op
class Gt2YoloTarget(BaseOperator):
    """
    Generate YOLOv3 targets by groud truth data, this operator is only used in
    fine grained YOLOv3 loss mode
    """

    def __init__(self,
                 anchors,
                 anchor_masks,
                 downsample_ratios,
                 num_classes=80,
                 iou_thresh=1.):
        super(Gt2YoloTarget, self).__init__()
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh

    def __call__(self, samples, context=None):
        assert len(self.anchor_masks) == len(self.downsample_ratios), \
            "anchor_masks', and 'downsample_ratios' should have same length."

        h, w = samples[0]['image'].shape[1:3]
        an_hw = np.array(self.anchors) / np.array([[w, h]])
        for sample in samples:

            # if 'gt_bbox' not in sample:
            #     sample['target0'] = None
            #     continue
            # if sample['gt_bbox'] is None or len(sample['gt_bbox']) == 0:
            #     sample['target0'] = None
            #     continue

            # im, gt_bbox, gt_class, gt_score = sample
            im = sample['image']
            gt_bbox = sample['gt_bbox']
            gt_class = sample['gt_class']
            if 'gt_score' not in sample:
                sample['gt_score'] = np.ones(
                    (gt_bbox.shape[0], 1), dtype=np.float32)
            gt_score = sample['gt_score']
            for i, (
                    mask, downsample_ratio
            ) in enumerate(zip(self.anchor_masks, self.downsample_ratios)):
                grid_h = int(h / downsample_ratio)
                grid_w = int(w / downsample_ratio)
                target = np.zeros(
                    (len(mask), 6 + self.num_classes, grid_h, grid_w),
                    dtype=np.float32)
                if sample["only_negimg"]:

                    # if 'gt_bbox' not in sample:
                    #     ""
                    # elif sample['gt_bbox'] is None or len(sample['gt_bbox']) == 0:
                    ""
                else:
                    for b in range(gt_bbox.shape[0]):
                        gx, gy, gw, gh = gt_bbox[b, :]
                        cls = gt_class[b]
                        score = gt_score[b]
                        if gw <= 0. or gh <= 0. or score <= 0.:
                            continue

                        # find best match anchor index
                        best_iou = 0.
                        best_idx = -1
                        for an_idx in range(an_hw.shape[0]):
                            iou = jaccard_overlap(
                                [0., 0., gw, gh],
                                [0., 0., an_hw[an_idx, 0], an_hw[an_idx, 1]])
                            if iou > best_iou:
                                best_iou = iou
                                best_idx = an_idx

                        gi = int(gx * grid_w)
                        gj = int(gy * grid_h)

                        # gtbox should be regresed in this layes if best match
                        # anchor index in anchor mask of this layer
                        if best_idx in mask:
                            best_n = mask.index(best_idx)

                            # x, y, w, h, scale
                            target[best_n, 0, gj, gi] = gx * grid_w - gi
                            target[best_n, 1, gj, gi] = gy * grid_h - gj
                            target[best_n, 2, gj, gi] = np.log(
                                gw * w / self.anchors[best_idx][0])
                            target[best_n, 3, gj, gi] = np.log(
                                gh * h / self.anchors[best_idx][1])
                            target[best_n, 4, gj, gi] = 2.0 - gw * gh

                            # objectness record gt_score
                            target[best_n, 5, gj, gi] = score

                            # classification
                            target[best_n, 6 + cls, gj, gi] = 1.

                        # For non-matched anchors, calculate the target if the iou
                        # between anchor and gt is larger than iou_thresh
                        if self.iou_thresh < 1:
                            for idx, mask_i in enumerate(mask):
                                if mask_i == best_idx: continue
                                iou = jaccard_overlap(
                                    [0., 0., gw, gh],
                                    [0., 0., an_hw[mask_i, 0], an_hw[mask_i, 1]])
                                if iou > self.iou_thresh and target[idx, 5, gj,
                                                                    gi] == 0.:
                                    # x, y, w, h, scale
                                    target[idx, 0, gj, gi] = gx * grid_w - gi
                                    target[idx, 1, gj, gi] = gy * grid_h - gj
                                    target[idx, 2, gj, gi] = np.log(
                                        gw * w / self.anchors[mask_i][0])
                                    target[idx, 3, gj, gi] = np.log(
                                        gh * h / self.anchors[mask_i][1])
                                    target[idx, 4, gj, gi] = 2.0 - gw * gh

                                    # objectness record gt_score
                                    target[idx, 5, gj, gi] = score

                                    # classification
                                    target[idx, 6 + cls, gj, gi] = 1.
                sample['target{}'.format(i)] = target

            # remove useless gt_class and gt_score after target calculated
            sample.pop('gt_class')
            sample.pop('gt_score')

        return samples


# #@register_op
class Gt2FCOSTarget(BaseOperator):
    """
    Generate FCOS targets by groud truth data
    """

    def __init__(self,
                 object_sizes_boundary,
                 center_sampling_radius,
                 downsample_ratios,
                 norm_reg_targets=False):
        super(Gt2FCOSTarget, self).__init__()
        self.center_sampling_radius = center_sampling_radius
        self.downsample_ratios = downsample_ratios
        self.INF = np.inf
        self.object_sizes_boundary = [-1] + object_sizes_boundary + [self.INF]
        object_sizes_of_interest = []
        for i in range(len(self.object_sizes_boundary) - 1):
            object_sizes_of_interest.append([
                self.object_sizes_boundary[i], self.object_sizes_boundary[i + 1]
            ])
        self.object_sizes_of_interest = object_sizes_of_interest
        self.norm_reg_targets = norm_reg_targets

    def _compute_points(self, w, h):
        """
        compute the corresponding points in each feature map
        :param h: image height
        :param w: image width
        :return: points from all feature map
        """
        locations = []
        for stride in self.downsample_ratios:
            shift_x = np.arange(0, w, stride).astype(np.float32)
            shift_y = np.arange(0, h, stride).astype(np.float32)
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            shift_x = shift_x.flatten()
            shift_y = shift_y.flatten()
            location = np.stack([shift_x, shift_y], axis=1) + stride // 2
            locations.append(location)
        num_points_each_level = [len(location) for location in locations]
        locations = np.concatenate(locations, axis=0)
        return locations, num_points_each_level

    def _convert_xywh2xyxy(self, gt_bbox, w, h):
        """
        convert the bounding box from style xywh to xyxy
        :param gt_bbox: bounding boxes normalized into [0, 1]
        :param w: image width
        :param h: image height
        :return: bounding boxes in xyxy style
        """
        bboxes = gt_bbox.copy()
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * w
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * h
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        return bboxes

    def _check_inside_boxes_limited(self, gt_bbox, xs, ys,
                                    num_points_each_level):
        """
        check if points is within the clipped boxes
        :param gt_bbox: bounding boxes
        :param xs: horizontal coordinate of points
        :param ys: vertical coordinate of points
        :return: the mask of points is within gt_box or not
        """
        bboxes = np.reshape(
            gt_bbox, newshape=[1, gt_bbox.shape[0], gt_bbox.shape[1]])
        bboxes = np.tile(bboxes, reps=[xs.shape[0], 1, 1])
        ct_x = (bboxes[:, :, 0] + bboxes[:, :, 2]) / 2
        ct_y = (bboxes[:, :, 1] + bboxes[:, :, 3]) / 2
        beg = 0
        clipped_box = bboxes.copy()
        for lvl, stride in enumerate(self.downsample_ratios):
            end = beg + num_points_each_level[lvl]
            stride_exp = self.center_sampling_radius * stride
            clipped_box[beg:end, :, 0] = np.maximum(
                bboxes[beg:end, :, 0], ct_x[beg:end, :] - stride_exp)
            clipped_box[beg:end, :, 1] = np.maximum(
                bboxes[beg:end, :, 1], ct_y[beg:end, :] - stride_exp)
            clipped_box[beg:end, :, 2] = np.minimum(
                bboxes[beg:end, :, 2], ct_x[beg:end, :] + stride_exp)
            clipped_box[beg:end, :, 3] = np.minimum(
                bboxes[beg:end, :, 3], ct_y[beg:end, :] + stride_exp)
            beg = end
        l_res = xs - clipped_box[:, :, 0]
        r_res = clipped_box[:, :, 2] - xs
        t_res = ys - clipped_box[:, :, 1]
        b_res = clipped_box[:, :, 3] - ys
        clipped_box_reg_targets = np.stack([l_res, t_res, r_res, b_res], axis=2)
        inside_gt_box = np.min(clipped_box_reg_targets, axis=2) > 0
        return inside_gt_box

    def __call__(self, samples, context=None):
        assert len(self.object_sizes_of_interest) == len(self.downsample_ratios), \
            "object_sizes_of_interest', and 'downsample_ratios' should have same length."

        for sample in samples:
            # im, gt_bbox, gt_class, gt_score = sample
            im = sample['image']
            bboxes = sample['gt_bbox']
            gt_class = sample['gt_class']
            # calculate the locations
            h, w = im.shape[1:3]
            points, num_points_each_level = self._compute_points(w, h)
            object_scale_exp = []
            for i, num_pts in enumerate(num_points_each_level):
                object_scale_exp.append(
                    np.tile(
                        np.array([self.object_sizes_of_interest[i]]),
                        reps=[num_pts, 1]))
            object_scale_exp = np.concatenate(object_scale_exp, axis=0)

            gt_area = (bboxes[:, 2] - bboxes[:, 0]) * (
                    bboxes[:, 3] - bboxes[:, 1])
            xs, ys = points[:, 0], points[:, 1]
            xs = np.reshape(xs, newshape=[xs.shape[0], 1])
            xs = np.tile(xs, reps=[1, bboxes.shape[0]])
            ys = np.reshape(ys, newshape=[ys.shape[0], 1])
            ys = np.tile(ys, reps=[1, bboxes.shape[0]])

            l_res = xs - bboxes[:, 0]
            r_res = bboxes[:, 2] - xs
            t_res = ys - bboxes[:, 1]
            b_res = bboxes[:, 3] - ys
            reg_targets = np.stack([l_res, t_res, r_res, b_res], axis=2)
            if self.center_sampling_radius > 0:
                is_inside_box = self._check_inside_boxes_limited(
                    bboxes, xs, ys, num_points_each_level)
            else:
                is_inside_box = np.min(reg_targets, axis=2) > 0
            # check if the targets is inside the corresponding level
            max_reg_targets = np.max(reg_targets, axis=2)
            lower_bound = np.tile(
                np.expand_dims(
                    object_scale_exp[:, 0], axis=1),
                reps=[1, max_reg_targets.shape[1]])
            high_bound = np.tile(
                np.expand_dims(
                    object_scale_exp[:, 1], axis=1),
                reps=[1, max_reg_targets.shape[1]])
            is_match_current_level = \
                (max_reg_targets > lower_bound) & \
                (max_reg_targets < high_bound)
            points2gtarea = np.tile(
                np.expand_dims(
                    gt_area, axis=0), reps=[xs.shape[0], 1])
            points2gtarea[is_inside_box == 0] = self.INF
            points2gtarea[is_match_current_level == 0] = self.INF
            points2min_area = points2gtarea.min(axis=1)
            points2min_area_ind = points2gtarea.argmin(axis=1)
            labels = gt_class[points2min_area_ind] + 1
            labels[points2min_area == self.INF] = 0
            reg_targets = reg_targets[range(xs.shape[0]), points2min_area_ind]
            ctn_targets = np.sqrt((reg_targets[:, [0, 2]].min(axis=1) / \
                                   reg_targets[:, [0, 2]].max(axis=1)) * \
                                  (reg_targets[:, [1, 3]].min(axis=1) / \
                                   reg_targets[:, [1, 3]].max(axis=1))).astype(np.float32)
            ctn_targets = np.reshape(
                ctn_targets, newshape=[ctn_targets.shape[0], 1])
            ctn_targets[labels <= 0] = 0
            pos_ind = np.nonzero(labels != 0)
            reg_targets_pos = reg_targets[pos_ind[0], :]
            split_sections = []
            beg = 0
            for lvl in range(len(num_points_each_level)):
                end = beg + num_points_each_level[lvl]
                split_sections.append(end)
                beg = end
            labels_by_level = np.split(labels, split_sections, axis=0)
            reg_targets_by_level = np.split(reg_targets, split_sections, axis=0)
            ctn_targets_by_level = np.split(ctn_targets, split_sections, axis=0)
            for lvl in range(len(self.downsample_ratios)):
                grid_w = int(np.ceil(w / self.downsample_ratios[lvl]))
                grid_h = int(np.ceil(h / self.downsample_ratios[lvl]))
                if self.norm_reg_targets:
                    sample['reg_target{}'.format(lvl)] = \
                        np.reshape(
                            reg_targets_by_level[lvl] / \
                            self.downsample_ratios[lvl],
                            newshape=[grid_h, grid_w, 4])
                else:
                    sample['reg_target{}'.format(lvl)] = np.reshape(
                        reg_targets_by_level[lvl],
                        newshape=[grid_h, grid_w, 4])
                sample['labels{}'.format(lvl)] = np.reshape(
                    labels_by_level[lvl], newshape=[grid_h, grid_w, 1])
                sample['centerness{}'.format(lvl)] = np.reshape(
                    ctn_targets_by_level[lvl], newshape=[grid_h, grid_w, 1])

            sample.pop('is_crowd')
            sample.pop('gt_class')
            sample.pop('gt_bbox')
        return samples


# #@register_op
class Gt2TTFTarget(BaseOperator):
    """
    Gt2TTFTarget
    Generate TTFNet targets by ground truth data
    """
    __shared__ = ['num_classes']
    """
    Gt2TTFTarget
    Generate TTFNet targets by ground truth data

    Args:
        num_classes(int): the number of classes.
        down_ratio(int): the down ratio from images to heatmap, 4 by default.
        alpha(float): the alpha parameter to generate gaussian target.
            0.54 by default.
    """

    def __init__(self, num_classes=80, down_ratio=4, alpha=0.54):
        super(Gt2TTFTarget, self).__init__()
        self.down_ratio = down_ratio
        self.num_classes = num_classes
        self.alpha = alpha

    def __call__(self, samples, context=None):
        output_size = samples[0]['image'].shape[1]
        feat_size = output_size // self.down_ratio
        for sample in samples:
            heatmap = np.zeros(
                (self.num_classes, feat_size, feat_size), dtype='float32')
            box_target = np.ones(
                (4, feat_size, feat_size), dtype='float32') * -1
            reg_weight = np.zeros((1, feat_size, feat_size), dtype='float32')

            gt_bbox = sample['gt_bbox']
            gt_class = sample['gt_class']

            bbox_w = gt_bbox[:, 2] - gt_bbox[:, 0] + 1
            bbox_h = gt_bbox[:, 3] - gt_bbox[:, 1] + 1
            area = bbox_w * bbox_h
            boxes_areas_log = np.log(area)
            boxes_ind = np.argsort(boxes_areas_log, axis=0)[::-1]
            boxes_area_topk_log = boxes_areas_log[boxes_ind]
            gt_bbox = gt_bbox[boxes_ind]
            gt_class = gt_class[boxes_ind]

            feat_gt_bbox = gt_bbox / self.down_ratio
            feat_gt_bbox = np.clip(feat_gt_bbox, 0, feat_size - 1)
            feat_hs, feat_ws = (feat_gt_bbox[:, 3] - feat_gt_bbox[:, 1],
                                feat_gt_bbox[:, 2] - feat_gt_bbox[:, 0])

            ct_inds = np.stack(
                [(gt_bbox[:, 0] + gt_bbox[:, 2]) / 2,
                 (gt_bbox[:, 1] + gt_bbox[:, 3]) / 2],
                axis=1) / self.down_ratio

            h_radiuses_alpha = (feat_hs / 2. * self.alpha).astype('int32')
            w_radiuses_alpha = (feat_ws / 2. * self.alpha).astype('int32')

            for k in range(len(gt_bbox)):
                cls_id = gt_class[k]
                fake_heatmap = np.zeros((feat_size, feat_size), dtype='float32')
                self.draw_truncate_gaussian(fake_heatmap, ct_inds[k],
                                            h_radiuses_alpha[k],
                                            w_radiuses_alpha[k])

                heatmap[cls_id] = np.maximum(heatmap[cls_id], fake_heatmap)
                box_target_inds = fake_heatmap > 0
                box_target[:, box_target_inds] = gt_bbox[k][:, None]

                local_heatmap = fake_heatmap[box_target_inds]
                ct_div = np.sum(local_heatmap)
                local_heatmap *= boxes_area_topk_log[k]
                reg_weight[0, box_target_inds] = local_heatmap / ct_div
            sample['ttf_heatmap'] = heatmap
            sample['ttf_box_target'] = box_target
            sample['ttf_reg_weight'] = reg_weight
            sample.pop('is_crowd')
            sample.pop('gt_class')
            sample.pop('gt_bbox')
            if 'gt_score' in sample:
                sample.pop('gt_score')
        return samples

    def draw_truncate_gaussian(self, heatmap, center, h_radius, w_radius):
        """

        :param heatmap:
        :param center:
        :param h_radius:
        :param w_radius:
        :return:
        """
        h, w = 2 * h_radius + 1, 2 * w_radius + 1
        sigma_x = w / 6
        sigma_y = h / 6
        gaussian = gaussian2D((h, w), sigma_x, sigma_y)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, w_radius), min(width - x, w_radius + 1)
        top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[h_radius - top:h_radius + bottom, w_radius -
                                                                     left:w_radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            heatmap[y - top:y + bottom, x - left:x + right] = np.maximum(
                masked_heatmap, masked_gaussian)
        return heatmap


# #@register_op
class Gt2Solov2Target(BaseOperator):
    """Assign mask target and labels in SOLOv2 network.
    Args:
        num_grids (list): The list of feature map grids size.
        scale_ranges (list): The list of mask boundary range.
        coord_sigma (float): The coefficient of coordinate area length.
        sampling_ratio (float): The ratio of down sampling.
    """

    def __init__(self,
                 num_grids=[40, 36, 24, 16, 12],
                 scale_ranges=[[1, 96], [48, 192], [96, 384], [192, 768],
                               [384, 2048]],
                 coord_sigma=0.2,
                 sampling_ratio=4.0):
        super(Gt2Solov2Target, self).__init__()
        self.num_grids = num_grids
        self.scale_ranges = scale_ranges
        self.coord_sigma = coord_sigma
        self.sampling_ratio = sampling_ratio

    def _scale_size(self, im, scale):
        h, w = im.shape[:2]
        new_size = (int(w * float(scale) + 0.5), int(h * float(scale) + 0.5))
        resized_img = cv2.resize(
            im, None, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        return resized_img

    def __call__(self, samples, context=None):
        sample_id = 0
        max_ins_num = [0] * len(self.num_grids)
        for sample in samples:
            gt_bboxes_raw = sample['gt_bbox']
            gt_labels_raw = sample['gt_class'] + 1
            im_c, im_h, im_w = sample['image'].shape[:]
            gt_masks_raw = sample['gt_segm'].astype(np.uint8)
            mask_feat_size = [
                int(im_h / self.sampling_ratio), int(im_w / self.sampling_ratio)
            ]
            gt_areas = np.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) *
                               (gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))
            ins_ind_label_list = []
            idx = 0
            for (lower_bound, upper_bound), num_grid \
                    in zip(self.scale_ranges, self.num_grids):

                hit_indices = ((gt_areas >= lower_bound) &
                               (gt_areas <= upper_bound)).nonzero()[0]
                num_ins = len(hit_indices)

                ins_label = []
                grid_order = []
                cate_label = np.zeros([num_grid, num_grid], dtype=np.int64)
                ins_ind_label = np.zeros([num_grid ** 2], dtype=np.bool)

                if num_ins == 0:
                    ins_label = np.zeros(
                        [1, mask_feat_size[0], mask_feat_size[1]],
                        dtype=np.uint8)
                    ins_ind_label_list.append(ins_ind_label)
                    sample['cate_label{}'.format(idx)] = cate_label.flatten()
                    sample['ins_label{}'.format(idx)] = ins_label
                    sample['grid_order{}'.format(idx)] = np.asarray(
                        [sample_id * num_grid * num_grid + 0], dtype=np.int32)
                    idx += 1
                    continue
                gt_bboxes = gt_bboxes_raw[hit_indices]
                gt_labels = gt_labels_raw[hit_indices]
                gt_masks = gt_masks_raw[hit_indices, ...]

                half_ws = 0.5 * (
                        gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.coord_sigma
                half_hs = 0.5 * (
                        gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.coord_sigma

                for seg_mask, gt_label, half_h, half_w in zip(
                        gt_masks, gt_labels, half_hs, half_ws):
                    if seg_mask.sum() == 0:
                        continue
                    # mass center
                    upsampled_size = (mask_feat_size[0] * 4,
                                      mask_feat_size[1] * 4)
                    center_h, center_w = ndimage.measurements.center_of_mass(
                        seg_mask)
                    coord_w = int(
                        (center_w / upsampled_size[1]) // (1. / num_grid))
                    coord_h = int(
                        (center_h / upsampled_size[0]) // (1. / num_grid))

                    # left, top, right, down
                    top_box = max(0,
                                  int(((center_h - half_h) / upsampled_size[0])
                                      // (1. / num_grid)))
                    down_box = min(num_grid - 1,
                                   int(((center_h + half_h) / upsampled_size[0])
                                       // (1. / num_grid)))
                    left_box = max(0,
                                   int(((center_w - half_w) / upsampled_size[1])
                                       // (1. / num_grid)))
                    right_box = min(num_grid - 1,
                                    int(((center_w + half_w) /
                                         upsampled_size[1]) // (1. / num_grid)))

                    top = max(top_box, coord_h - 1)
                    down = min(down_box, coord_h + 1)
                    left = max(coord_w - 1, left_box)
                    right = min(right_box, coord_w + 1)

                    cate_label[top:(down + 1), left:(right + 1)] = gt_label
                    seg_mask = self._scale_size(
                        seg_mask, scale=1. / self.sampling_ratio)
                    for i in range(top, down + 1):
                        for j in range(left, right + 1):
                            label = int(i * num_grid + j)
                            cur_ins_label = np.zeros(
                                [mask_feat_size[0], mask_feat_size[1]],
                                dtype=np.uint8)
                            cur_ins_label[:seg_mask.shape[0], :seg_mask.shape[
                                1]] = seg_mask
                            ins_label.append(cur_ins_label)
                            ins_ind_label[label] = True
                            grid_order.append(sample_id * num_grid * num_grid +
                                              label)
                if ins_label == []:
                    ins_label = np.zeros(
                        [1, mask_feat_size[0], mask_feat_size[1]],
                        dtype=np.uint8)
                    ins_ind_label_list.append(ins_ind_label)
                    sample['cate_label{}'.format(idx)] = cate_label.flatten()
                    sample['ins_label{}'.format(idx)] = ins_label
                    sample['grid_order{}'.format(idx)] = np.asarray(
                        [sample_id * num_grid * num_grid + 0], dtype=np.int32)
                else:
                    ins_label = np.stack(ins_label, axis=0)
                    ins_ind_label_list.append(ins_ind_label)
                    sample['cate_label{}'.format(idx)] = cate_label.flatten()
                    sample['ins_label{}'.format(idx)] = ins_label
                    sample['grid_order{}'.format(idx)] = np.asarray(
                        grid_order, dtype=np.int32)
                    assert len(grid_order) > 0
                max_ins_num[idx] = max(
                    max_ins_num[idx],
                    sample['ins_label{}'.format(idx)].shape[0])
                idx += 1
            ins_ind_labels = np.concatenate([
                ins_ind_labels_level_img
                for ins_ind_labels_level_img in ins_ind_label_list
            ])
            fg_num = np.sum(ins_ind_labels)
            sample['fg_num'] = fg_num
            sample_id += 1

            sample.pop('is_crowd')
            sample.pop('gt_class')
            sample.pop('gt_bbox')
            sample.pop('gt_poly')
            sample.pop('gt_segm')

        # padding batch
        for data in samples:
            for idx in range(len(self.num_grids)):
                gt_ins_data = np.zeros(
                    [
                        max_ins_num[idx],
                        data['ins_label{}'.format(idx)].shape[1],
                        data['ins_label{}'.format(idx)].shape[2]
                    ],
                    dtype=np.uint8)
                gt_ins_data[0:data['ins_label{}'.format(idx)].shape[
                    0], :, :] = data['ins_label{}'.format(idx)]
                gt_grid_order = np.zeros([max_ins_num[idx]], dtype=np.int32)
                gt_grid_order[0:data['grid_order{}'.format(idx)].shape[
                    0]] = data['grid_order{}'.format(idx)]
                data['ins_label{}'.format(idx)] = gt_ins_data
                data['grid_order{}'.format(idx)] = gt_grid_order

        return samples


# #@register_op
class RboxPadBatch(BaseOperator):
    """
    Pad a batch of samples so they can be divisible by a stride.
    The layout of each image should be 'CHW'. And convert poly to rbox.
    Args:
        pad_to_stride (int): If `pad_to_stride > 0`, pad zeros to ensure
            height and width is divisible by `pad_to_stride`.
    """

    def __init__(self, pad_to_stride=0, pad_gt=False):
        super(RboxPadBatch, self).__init__()
        self.pad_to_stride = pad_to_stride
        self.pad_gt = pad_gt

    def __call__(self, samples, context=None):
        """
        Args:
            samples (list): a batch of sample, each is dict.
        """
        coarsest_stride = self.pad_to_stride

        max_shape = np.array([data['image'].shape for data in samples]).max(
            axis=0)
        if coarsest_stride > 0:
            max_shape[1] = int(
                np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
            max_shape[2] = int(
                np.ceil(max_shape[2] / coarsest_stride) * coarsest_stride)

        for data in samples:
            im = data['image']
            im_c, im_h, im_w = im.shape[:]
            padding_im = np.zeros(
                (im_c, max_shape[1], max_shape[2]), dtype=np.float32)
            padding_im[:, :im_h, :im_w] = im
            data['image'] = padding_im
            if 'semantic' in data and data['semantic'] is not None:
                semantic = data['semantic']
                padding_sem = np.zeros(
                    (1, max_shape[1], max_shape[2]), dtype=np.float32)
                padding_sem[:, :im_h, :im_w] = semantic
                data['semantic'] = padding_sem
            if 'gt_segm' in data and data['gt_segm'] is not None:
                gt_segm = data['gt_segm']
                padding_segm = np.zeros(
                    (gt_segm.shape[0], max_shape[1], max_shape[2]),
                    dtype=np.uint8)
                padding_segm[:, :im_h, :im_w] = gt_segm
                data['gt_segm'] = padding_segm
        if self.pad_gt:
            gt_num = []
            if 'gt_poly' in data and data['gt_poly'] is not None and len(data[
                                                                             'gt_poly']) > 0:
                pad_mask = True
            else:
                pad_mask = False

            if pad_mask:
                poly_num = []
                poly_part_num = []
                point_num = []
            for data in samples:
                gt_num.append(data['gt_bbox'].shape[0])
                if pad_mask:
                    poly_num.append(len(data['gt_poly']))
                    for poly in data['gt_poly']:
                        poly_part_num.append(int(len(poly)))
                        for p_p in poly:
                            point_num.append(int(len(p_p) / 2))
            gt_num_max = max(gt_num)

            for i, sample in enumerate(samples):
                assert 'gt_rbox' in sample
                assert 'gt_rbox2poly' in sample
                gt_box_data = -np.ones([gt_num_max, 4], dtype=np.float32)
                gt_class_data = -np.ones([gt_num_max], dtype=np.int32)
                is_crowd_data = np.ones([gt_num_max], dtype=np.int32)

                if pad_mask:
                    poly_num_max = max(poly_num)
                    poly_part_num_max = max(poly_part_num)
                    point_num_max = max(point_num)
                    gt_masks_data = -np.ones(
                        [poly_num_max, poly_part_num_max, point_num_max, 2],
                        dtype=np.float32)

                gt_num = sample['gt_bbox'].shape[0]
                gt_box_data[0:gt_num, :] = sample['gt_bbox']
                gt_class_data[0:gt_num] = np.squeeze(sample['gt_class'])
                is_crowd_data[0:gt_num] = np.squeeze(sample['is_crowd'])
                if pad_mask:
                    for j, poly in enumerate(sample['gt_poly']):
                        for k, p_p in enumerate(poly):
                            pp_np = np.array(p_p).reshape(-1, 2)
                            gt_masks_data[j, k, :pp_np.shape[0], :] = pp_np
                    sample['gt_poly'] = gt_masks_data
                sample['gt_bbox'] = gt_box_data
                sample['gt_class'] = gt_class_data
                sample['is_crowd'] = is_crowd_data
                # ploy to rbox
                polys = sample['gt_rbox2poly']
                rbox = bbox_utils.poly_to_rbox(polys)
                sample['gt_rbox'] = rbox

        return samples
