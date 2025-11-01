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
from __future__ import absolute_import, division, print_function
import os
import hashlib

import paddle
import paddle.nn as nn
from paddle.nn import AdaptiveAvgPool2D, Conv2D, Dropout, Linear, BatchNorm2D

import numpy as np


def make_divisible(v, divisor=8, min_val=None):
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _create_act(act):
    if act == "hardswish":
        return nn.Hardswish()
    elif act == "relu":
        return nn.ReLU()
    elif act is None:
        return None
    else:
        raise RuntimeError(
            "The activation function is not supported: {}".format(act))


class StemBlock(nn.Layer):
    def __init__(self, inplanes=3, outplanes=16, kernel_size=3, stride=2, padding=1, num_groups=1, if_act=True, act="hardswish", bias_attr=False):
        super().__init__()

        self.conv = Conv2D(
            in_channels=inplanes,
            out_channels=outplanes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            bias_attr=False)
        self.bn = BatchNorm2D(outplanes, epsilon=0.001)
        self.if_act = if_act
        self.act = nn.Hardswish()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            x = self.act(x)
        return x


class Tail_Squeeze_Block(nn.Layer):
    def __init__(self, inplanes=160, outplanes=960, kernel_size=1, stride=1, padding=0, num_groups=1, if_act=True, act="hardswish", bias_attr=False):
        super().__init__()

        self.conv = Conv2D(
            in_channels=inplanes,
            out_channels=outplanes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            bias_attr=False)
        self.bn = BatchNorm2D(outplanes, epsilon=0.001)
        self.if_act = if_act
        self.act = nn.Hardswish()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            x = self.act(x)
        return x


class HeadBlock(nn.Layer):
    def __init__(self, 
    inplanes=960, outplanes=1280 ,num_classes=1000, kernel_size=1, stride=1, padding=0,
    dropout_prob=0.1):
        super().__init__()

        self.avg_pool = AdaptiveAvgPool2D(1)
        self.conv = Linear(inplanes, outplanes, bias_attr=True)
        self.hardswish = nn.Hardswish()
        self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)
        # self.drop = nn.Dropout(p=dropout_prob, mode="downscale_in_infer")
        self.fc = Linear(outplanes, num_classes, bias_attr=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.conv(x)
        x = self.hardswish(x)
        # x = self.drop(x)
        x = self.fc(x)
        return x


class ResidualUnit(nn.Layer):
    def __init__(self,
                 in_c,
                 mid_c,
                 out_c,
                 filter_size,
                 stride,
                 use_se,
                 act=None):
        super().__init__()
        self.if_shortcut = stride == 1 and in_c == out_c
        self.if_se = use_se
        self.in_c = in_c
        self.mid_c = mid_c
        
        if self.in_c != self.mid_c:
            self.invert_conv1 = Conv2D(
                in_channels=in_c,
                out_channels=mid_c,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias_attr=False)
            self.invert_bn1 = BatchNorm2D(mid_c, epsilon=0.001)

        self.depth_conv2 = Conv2D(
            in_channels=mid_c,
            out_channels=mid_c,
            kernel_size=filter_size,
            stride=stride,
            padding=int((filter_size - 1) // 2),
            groups=mid_c,
            bias_attr=False)
        self.depth_bn2 = BatchNorm2D(mid_c, epsilon=0.001)
        if self.if_se:
            self.mid_se = SEModule(mid_c)

        self.point_conv3 = Conv2D(
            in_channels=mid_c,
            out_channels=out_c,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias_attr=False)
        self.point_bn3 = BatchNorm2D(out_c, epsilon=0.001)
        self.act = _create_act(act)

    def forward(self, x):
        identity = x
        if self.in_c != self.mid_c:
            x = self.invert_conv1(x)
            x = self.invert_bn1(x)
            x = self.act(x)
        x = self.depth_conv2(x)
        x = self.depth_bn2(x)
        x = self.act(x)

        if self.if_se:
            x = self.mid_se(x)

        x = self.point_conv3(x)
        x = self.point_bn3(x)
        if self.if_shortcut:
            x = paddle.add(identity, x)
        return x


class SEModule(nn.Layer):
    def __init__(self, channel, reduction=4):
        super().__init__()
        num_mid = make_divisible(channel // reduction)
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.con_v1 = Conv2D(
            in_channels=channel,
            out_channels=num_mid,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=True)
        self.relu = nn.ReLU()
        self.con_v2 = Conv2D(
            in_channels=num_mid,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=True)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.con_v1(x)
        x = self.relu(x)
        x = self.con_v2(x)
        x = self.hardsigmoid(x)
        return paddle.multiply(x=identity, y=x)


class Model(nn.Layer):
    def __init__(self, arch, 
        config={'i': [224],
                'd': [(2, 3), (2, 3), (2, 3), (2, 3), (2, 3)], 
                'e': {2: 3, 3: 4, 4: 4.5, 5: 5.2, 6: 6}}, 
        scale=1.0,
        inplanes=16, 
        class_squeeze=576,
        class_expand=1024,
        class_num=1000,
        dropout_prob=0.2):
        super(Model,self).__init__()

        self.scale = scale
        self.inplanes = make_divisible(int(inplanes * scale), 8)
        self.class_squeeze = make_divisible(int(class_squeeze * scale), 8)
        self.class_expand = make_divisible(int(class_expand * scale), 8)
        self.class_num = class_num
        self.dropout_prob = dropout_prob
        
        self.im_size_dict = {i: x for i, x in enumerate(config['i'], 1)}
        self.depth_dict = {k: k for s, e in config['d'] for k in range(s, e+1)}
        self.arch = arch

        im_size_code = arch[0]
        depth_code = arch[1:5]      
        new_layers = [int(i) for idx,i in enumerate(depth_code)]
        kernel_code = arch[5:17]
        expand_code = arch[17:]

        self.im_size = self.im_size_dict[int(im_size_code)]
        self.depth_list = [int(x) for x in depth_code]

        base_stage_width = [16, 16, 24, 40, 48, 96]

        stride_stages = [2, 2, 2, 1, 2]
        act_stages = ['relu', 'relu', 'hardswish', 'hardswish', 'hardswish']
        se_stages = [True, False, True, True, True]
        n_block_list = [1] + new_layers

        width_list = []
        for base_width in base_stage_width:
            width = make_divisible(base_width * scale, 8)
            width_list.append(width)

        input_channel, first_block_dim = width_list[0], width_list[1]

        first_block = ResidualUnit(
                in_c = input_channel,
                mid_c = input_channel,
                out_c = first_block_dim,
                filter_size=3,
                stride=stride_stages[0],
                use_se=se_stages[0],
                act=act_stages[0])

        # inverted residual blocks
        self.block_group_info = []
        self.blocks = nn.LayerList([StemBlock(3, input_channel, 3, 2, 1, 1, True, "hardswish",False)])
        self.blocks.append(first_block)
        _block_index = 2
        feature_dim = first_block_dim

        l = len(new_layers)
        j = 0
        for width, n_block, s, act_func, use_se in zip(width_list[2:], n_block_list[1:],
                                                stride_stages[1:], act_stages[1:], se_stages[1:]):
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                fs = int(kernel_code[3*j : 3*(j+1)][i])
                ex = int(expand_code[3*j : 3*(j+1)][i])
                assert fs > 0, "filter size > 0"
                assert ex > 0, "expand ratio > 0"
                self.blocks.append(ResidualUnit(
                        in_c = feature_dim,
                        mid_c = make_divisible(int(feature_dim * config['e'][ex]), 8),
                        out_c = output_channel,
                        filter_size = fs,
                        stride = stride,
                        use_se = use_se,
                        act = act_func))
                feature_dim = output_channel
            j += 1
        self.blocks.append(Tail_Squeeze_Block(width_list[-1], self.class_squeeze, 1, 1, 0, 1, True, "hardswish", False))
        self.blocks.append(HeadBlock(self.class_squeeze, self.class_expand, self.class_num, 1, 1, 0, self.dropout_prob))
        self.act_depth_list = new_layers

    def forward(self, x):
        x = self.blocks[0](x)
        x = self.blocks[1](x)
        # blocks
        outputs = []

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.act_depth_list[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x = self.blocks[idx](x)

            if stage_id in [0, 2, 3]:
                outputs.append(x)
        return outputs
        # x = self.blocks[-2](x)
        # x = self.blocks[-1](x)
        # return x

    # def export(self, dir_path, size=None):
    #     if size is not None:
    #         im_size = size
    #     else:
    #         im_size = self.im_size
    #     dir_name = hashlib.md5(self.arch.encode(encoding='UTF-8')).hexdigest()
    #     save_path = os.path.join(dir_path, dir_name)
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     else:
    #         print('model dir exists...')
    #     with open(os.path.join(save_path, 'code.txt'), 'w') as f:
    #         f.write('{}'.format(self.arch))
    #     export_static_model(self, os.path.join(save_path, 'inference'), im_size)


