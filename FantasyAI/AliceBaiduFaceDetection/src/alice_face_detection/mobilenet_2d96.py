from __future__ import absolute_import, division, print_function
import paddle
import paddle.nn as nn
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
    def __init__(
            self,
            inplanes=3,
            outplanes=16,
            kernel_size=3,
            stride=2,
            padding=1,
            num_groups=1,
            if_act=True,
            act="hardswish",
            bias_attr=False
    ):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels=inplanes,
            out_channels=outplanes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            bias_attr=False)
        self.bn = nn.BatchNorm2D(outplanes)
        self.if_act = if_act
        self.act = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            x = self.act(x)
        return x


class Tail_Squeeze_Block(nn.Layer):
    def __init__(
            self,
            inplanes=160,
            outplanes=960,
            kernel_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=True,
            act="hardswish",
            bias_attr=False
    ):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels=inplanes,
            out_channels=outplanes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            bias_attr=False)
        self.bn = nn.BatchNorm2D(outplanes)
        self.if_act = if_act
        self.act = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            x = self.act(x)
        return x


class HeadBlock(nn.Layer):
    def __init__(
            self,
            inplanes=960,
            outplanes=1280,
            num_classes=1000,
            kernel_size=1,
            stride=1,
            padding=0,
            dropout_prob=0.1
    ):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv = nn.Conv2D(
            in_channels=inplanes,
            out_channels=outplanes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=False)
        self.hardswish = nn.Hardswish()
        self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)
        # self.drop = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(outplanes, num_classes, bias_attr=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.hardswish(x)
        x = self.flatten(x)
        # x = self.drop(x)
        x = self.fc(x)

        return x


class ResidualUnit(nn.Layer):
    def __init__(
            self,
            in_c,
            mid_c,
            out_c,
            filter_size,
            stride,
            use_se,
            act=None
    ):
        super().__init__()
        self.if_shortcut = stride == 1 and in_c == out_c
        self.if_se = use_se
        self.in_c = in_c
        self.mid_c = mid_c

        if self.in_c != self.mid_c:
            self.invert_conv1 = nn.Conv2D(
                in_channels=in_c,
                out_channels=mid_c,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias_attr=False)
            self.invert_bn1 = nn.BatchNorm2D(mid_c)

        self.depth_conv2 = nn.Conv2D(
            in_channels=mid_c,
            out_channels=mid_c,
            kernel_size=filter_size,
            stride=stride,
            padding=int((filter_size - 1) // 2),
            groups=mid_c,
            bias_attr=False)
        self.depth_bn2 = nn.BatchNorm2D(mid_c)

        self.point_conv3 = nn.Conv2D(
            in_channels=mid_c,
            out_channels=out_c,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias_attr=False)
        self.point_bn3 = nn.BatchNorm2D(out_c)
        self.act = _create_act(act)

        if self.if_se:
            self.mid_se = SEModule(mid_c)

    def forward(self, x):
        identity = x
        if self.in_c != self.mid_c:
            # print(x)
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
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.con_v1 = nn.Conv2D(
            in_channels=channel,
            out_channels=num_mid,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=True)
        self.relu = nn.ReLU()
        self.con_v2 = nn.Conv2D(
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
    def __init__(
            self,
            arch,
            config={'i': [224], 'd': [(2, 4), (2, 4), (2, 4), (2, 4), (2, 4)]},
            inplanes=16,
            class_squeeze=960,
            class_expand=1280,
            class_num=1000,
            dropout_prob=0.1):
        super(Model, self).__init__()

        assert arch[1] in ['1', '2']

        im_size_code = arch[0]
        scale = 1.0 if arch[1] == '1' else 1.2
        depth_code = arch[2:7]
        new_layers = [int(i) for idx, i in enumerate(depth_code)]
        kernel_code = arch[7:27]
        expand_code = arch[27:]

        self.inplanes = make_divisible(int(inplanes * scale), 8)
        self.class_squeeze = make_divisible(int(class_squeeze * scale), 8)
        self.class_expand = make_divisible(int(class_expand * scale), 8)
        self.class_num = class_num
        self.dropout_prob = dropout_prob

        self.im_size_dict = {i: x for i, x in enumerate(config['i'], 1)}
        self.depth_dict = {k: k for s, e in config['d'] for k in range(s, e + 1)}

        self.im_size = self.im_size_dict[int(im_size_code)]
        self.depth_list = [int(x) for x in depth_code]

        base_stage_width = [16, 16, 24, 40, 80, 112, 160]

        stride_stages = [1, 2, 2, 2, 1, 2]
        act_stages = ['relu', 'relu', 'relu', "hardswish", "hardswish", "hardswish"]
        se_stages = [False, False, True, False, True, True]
        n_block_list = [1] + new_layers

        width_list = []
        for base_width in base_stage_width:
            width = make_divisible(base_width * scale, 8)
            width_list.append(width)

        input_channel, first_block_dim = width_list[0], width_list[1]

        first_block = ResidualUnit(
            in_c=input_channel,
            mid_c=input_channel,
            out_c=first_block_dim,
            filter_size=3,
            stride=1,
            use_se=False,
            act="relu")

        # inverted residual blocks
        self.block_group_info = []
        self.blocks = nn.LayerList([StemBlock(3, input_channel, 3, 2, 1, 1, True, "hardswish", False)])
        self.blocks.append(first_block)
        _block_index = 2
        feature_dim = first_block_dim

        l = len(new_layers)
        j = 0
        for width, n_block, s, act_func, use_se in zip(
                width_list[2:], n_block_list[1:], stride_stages[1:], act_stages[1:], se_stages[1:]):
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                fs = int(kernel_code[4 * j:4 * (j + 1)][i])
                ex = int(expand_code[4 * j:4 * (j + 1)][i])
                assert fs > 0, "filter size > 0"
                assert ex > 0, "expand ratio > 0"
                self.blocks.append(
                    ResidualUnit(
                        in_c=feature_dim,
                        mid_c=feature_dim * ex,
                        out_c=output_channel,
                        filter_size=fs,
                        stride=stride,
                        use_se=use_se,
                        act=act_func)
                )
                feature_dim = output_channel
            j += 1
        self.blocks.append(Tail_Squeeze_Block(width_list[-1], self.class_squeeze, 1, 1, 0, 1, True, "hardswish", False))
        self.blocks.append(HeadBlock(self.class_squeeze, self.class_expand, self.class_num, 1, 1, 0, self.dropout_prob))
        self.act_depth_list = new_layers

    def forward(self, x):
        # print("x:", x.shape)
        x = self.blocks[0](x)
        # print("x:", x.shape)
        x = self.blocks[1](x)
        # print("x:", x.shape)
        # blocks

        outputs = []

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.act_depth_list[stage_id]
            active_idx = block_idx[:depth]
            # print ("stage_id", stage_id, "depth",depth, "active_idx",active_idx)
            for idx in active_idx:
                x = self.blocks[idx](x)
                # print("x:",x.shape)

            if stage_id in  [1, 3, 4]:
                outputs.append(x)
        return outputs
        x = self.blocks[-2](x)
        print ("x:",x.shape)
        x = self.blocks[-1](x)
        print ("x:",x.shape)
        return x