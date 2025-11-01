"""
Paddle for ppyolo
backbone=MobileNet V3
neck=PPYOLO
head=YOLOV3
"""
import paddle
import mobilenet_v3
import mobilenet_2d96
import mobilenet_2d4f5904
import mobilenet_50M224
import mobilenet_c7230302
import mobilenet_792bcf52
import mobilenet_b0b5e552
import x2paddle_code_2d4f5904_prune50
# import x2paddle_code
import yolo_fpn
import yolo_head
import yolo_loss
import iou_loss
import post_process

from layers import YOLOBox, MultiClassNMS
# import cv2
import numpy as np


class PPYoloTiny(paddle.nn.Layer):
    """PPYolo. based MobileNetV3Tiny"""

    def __init__(self, model="mbv3"):
        super(PPYoloTiny, self).__init__()
        num_classes = 1
        self.model = model

        if self.model == "2d96":
            self.backbone = mobilenet_2d96.Model(#feature_maps=[7, 13, 16],
                                                 arch='12222333300530033003330335033006400640063303640')
            self.neck = yolo_fpn.PPYOLOTinyFPN(#in_channels=[24, 56, 80],
                                               in_channels=[48,136,192],
                                               detection_block_channels=[160, 128, 96],
                                               coord_conv=True, spp=True, drop_block=True)
            self.yolo_head = yolo_head.YOLOv3Head(in_channels=[160, 128, 96],
                                                  anchors=[[10, 15], [24, 36], [72, 42],
                                                           [35, 87], [102, 96], [60, 170],
                                                           [220, 125], [128, 222], [264, 266]],
                                                  anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                                                  num_classes=num_classes,
                                                  loss=yolo_loss.YOLOv3Loss(num_classes=num_classes, ignore_thresh=0.5,
                                                                            label_smooth=False, downsample=[32, 16, 8],
                                                                            scale_x_y=1.05,
                                                                            iou_loss=iou_loss.IouLoss()),
                                                  )

        elif self.model == '2d4f5904':
            # backbone
            self.backbone = mobilenet_2d4f5904.Model(arch='1222232323000000233300000045550000000000006363000000', block='basic')
            self.neck = yolo_fpn.PPYOLOTinyFPN(in_channels=[112, 208, 464], detection_block_channels=[160, 128, 96],
                                               coord_conv=True, spp=True, drop_block=True)
            self.yolo_head = yolo_head.YOLOv3Head(in_channels=[160, 128, 96],
                                                  anchors=[[10, 15], [24, 36], [72, 42],
                                                           [35, 87], [102, 96], [60, 170],
                                                           [220, 125], [128, 222], [264, 266]],
                                                  anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                                                  num_classes=num_classes,
                                                  loss=yolo_loss.YOLOv3Loss(num_classes=num_classes, ignore_thresh=0.5,
                                                                            label_smooth=False, downsample=[32, 16, 8],
                                                                            scale_x_y=1.05,
                                                                            iou_loss=iou_loss.IouLoss()))

        elif self.model == 'model50M224':
            self.backbone = mobilenet_50M224.Model(arch='12223330550550555420350220666')
            self.neck = yolo_fpn.PPYOLOTinyFPN(in_channels=[24, 48, 96], detection_block_channels=[160, 128, 96],
                                               coord_conv=True, spp=True, drop_block=True)
            self.yolo_head = yolo_head.YOLOv3Head(in_channels=[160, 128, 96],
                                                  anchors=[[10, 15], [24, 36], [72, 42],
                                                           [35, 87], [102, 96], [60, 170],
                                                           [220, 125], [128, 222], [264, 266]],
                                                  anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                                                  num_classes=num_classes,
                                                  loss=yolo_loss.YOLOv3Loss(num_classes=num_classes, ignore_thresh=0.5,
                                                                            label_smooth=False, downsample=[32, 16, 8],
                                                                            scale_x_y=1.05,
                                                                            iou_loss=iou_loss.IouLoss()))

        elif self.model == '792bcf52':
            self.backbone = mobilenet_792bcf52.Model(arch='11222233300530035005300553034003400330033006660')
            self.neck = yolo_fpn.PPYOLOTinyFPN(in_channels=[40, 112, 160], detection_block_channels=[160, 128, 96],
                                               coord_conv=True, spp=True, drop_block=True)
            self.yolo_head = yolo_head.YOLOv3Head(in_channels=[160, 128, 96],
                                                  anchors=[[10, 15], [24, 36], [72, 42],
                                                           [35, 87], [102, 96], [60, 170],
                                                           [220, 125], [128, 222], [264, 266]],
                                                  anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                                                  num_classes=num_classes,
                                                  loss=yolo_loss.YOLOv3Loss(num_classes=num_classes, ignore_thresh=0.5,
                                                                            label_smooth=False, downsample=[32, 16, 8],
                                                                            scale_x_y=1.05,
                                                                            iou_loss=iou_loss.IouLoss()))
        elif self.model == 'b0b5e552':
            self.backbone = mobilenet_792bcf52.Model(arch='11222235300350033005500353033006300330043004440')
            self.neck = yolo_fpn.PPYOLOTinyFPN(in_channels=[40, 112, 160], detection_block_channels=[160, 128, 96],
                                               coord_conv=True, spp=True, drop_block=True)
            self.yolo_head = yolo_head.YOLOv3Head(in_channels=[160, 128, 96],
                                                  anchors=[[10, 15], [24, 36], [72, 42],
                                                           [35, 87], [102, 96], [60, 170],
                                                           [220, 125], [128, 222], [264, 266]],
                                                  anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                                                  num_classes=num_classes,
                                                  loss=yolo_loss.YOLOv3Loss(num_classes=num_classes, ignore_thresh=0.5,
                                                                            label_smooth=False, downsample=[32, 16, 8],
                                                                            scale_x_y=1.05,
                                                                            iou_loss=iou_loss.IouLoss()))

        elif self.model == 'c7230302':
            self.backbone = mobilenet_792bcf52.Model(arch='11222223500330053005500550034003300640044003300')
            self.neck = yolo_fpn.PPYOLOTinyFPN(in_channels=[40, 112, 160], detection_block_channels=[160, 128, 96],
                                               coord_conv=True, spp=True, drop_block=True)
            self.yolo_head = yolo_head.YOLOv3Head(in_channels=[160, 128, 96],
                                                  anchors=[[10, 15], [24, 36], [72, 42],
                                                           [35, 87], [102, 96], [60, 170],
                                                           [220, 125], [128, 222], [264, 266]],
                                                  anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                                                  num_classes=num_classes,
                                                  loss=yolo_loss.YOLOv3Loss(num_classes=num_classes, ignore_thresh=0.5,
                                                                            label_smooth=False, downsample=[32, 16, 8],
                                                                            scale_x_y=1.05,
                                                                            iou_loss=iou_loss.IouLoss()))

        elif self.model == 'mbv3':
            self.backbone = mobilenet_v3.MobileNetV3(model_name="large", scale=0.5, with_extra_blocks=False,
                                                     extra_block_filters=[], feature_maps=[7, 13, 16])
            self.neck = yolo_fpn.PPYOLOTinyFPN(in_channels=[24, 56, 80], detection_block_channels=[160, 128, 96],
                                               coord_conv=True, spp=True, drop_block=True)
            self.yolo_head = yolo_head.YOLOv3Head(in_channels=[160, 128, 96],
                                                  anchors=[[10, 15], [24, 36], [72, 42],
                                                           [35, 87], [102, 96], [60, 170],
                                                           [220, 125], [128, 222], [264, 266]],
                                                  anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                                                  num_classes=num_classes,
                                                  loss=yolo_loss.YOLOv3Loss(num_classes=num_classes, ignore_thresh=0.5,
                                                                            label_smooth=False, downsample=[32, 16, 8],
                                                                            scale_x_y=1.05,
                                                                            iou_loss=iou_loss.IouLoss()))
        elif self.model == 'onnx_2d4f5904':
            self.backbone = x2paddle_code_2d4f5904_prune50.ONNXModel()
            # self.neck = yolo_fpn.PPYOLOTinyFPN(in_channels=[112, 208, 464], detection_block_channels=[160, 128, 96],
            #                                    coord_conv=True, spp=True, drop_block=True)
            self.yolo_head = yolo_head.YOLOv3Head(in_channels=[160, 128, 96],
                                                  anchors=[[10, 15], [24, 36], [72, 42],
                                                           [35, 87], [102, 96], [60, 170],
                                                           [220, 125], [128, 222], [264, 266]],
                                                  anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                                                  num_classes=num_classes,
                                                  loss=yolo_loss.YOLOv3Loss(num_classes=num_classes, ignore_thresh=0.5,
                                                                            label_smooth=False, downsample=[32, 16, 8],
                                                                            scale_x_y=1.05,
                                                                            iou_loss=iou_loss.IouLoss()))

        #         self.post_process = post_process.BBoxPostProcess(decode=YOLOBox(num_classes=num_classes, conf_thresh=0.005,
#                                                                         downsample_ratio=32, clip_bbox=True,
#                                                                         scale_x_y=1.05), num_classes=num_classes,
#                                                          nms=MultiClassNMS(keep_top_k=100, nms_threshold=0.45,
#                                                                            nms_top_k=1000, score_threshold=0.005))
        
#         elif self.model == 'onnx_2d4f5904_1':
#             self.backbone = x2paddle_code.ONNXModel()
#             self.yolo_head = yolo_head.YOLOv3Head(in_channels=[160, 128, 96],
#                                                   anchors=[[10, 15], [24, 36], [72, 42],
#                                                            [35, 87], [102, 96], [60, 170],
#                                                            [220, 125], [128, 222], [264, 266]],
#                                                   anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
#                                                   num_classes=num_classes,
#                                                   loss=yolo_loss.YOLOv3Loss(num_classes=num_classes, ignore_thresh=0.5,
#                                                                             label_smooth=False, downsample=[32, 16, 8],
#                                                                             scale_x_y=1.05,
#                                                                             iou_loss=iou_loss.IouLoss()))

        self.post_process = post_process.BBoxPostProcess(decode=YOLOBox(num_classes=num_classes, conf_thresh=0.005,
                                                                        downsample_ratio=32, clip_bbox=True,
                                                                        scale_x_y=1.05), num_classes=num_classes,
                                                         nms=MultiClassNMS(keep_top_k=100, nms_threshold=0.45,
                                                                           nms_top_k=1000, score_threshold=0.005))
    # from paddle.static import InputSpec
    # @paddle.jit.to_static(input_spec=[InputSpec(shape=[None, 3, 160, 288])])
    def forward(self, inputs, targets=None):
        """
        net forward
        :param inputs:inputs
        :param targets:labels
        :return:
        """

        ##原始结构
        out = self.backbone(inputs)
        # return out
        # for item in out:
        #     print ("item:", item.shape)
        out = self.neck(out)

        if targets is not None:
            return self.yolo_head(out, targets)
        else:
            yolo_head_outs = self.yolo_head(out)
            # return yolo_head_outs

            inputs_shape = np.zeros((inputs.shape[0], 2))
            for i in range(inputs.shape[0]):
                inputs_shape[i, 0] = inputs.shape[2]
                inputs_shape[i, 1] = inputs.shape[3]

            bbox, bbox_num = self.post_process(
                yolo_head_outs, self.yolo_head.mask_anchors,
                paddle.to_tensor(inputs_shape), 1.)

            return bbox, bbox_num
        return out



        # #融合bn融合首层
        # out = self.backbone(inputs)
        # if targets is not None:
        #
        #     return self.yolo_head(out, targets)
        #     # return out
        # else:
        #     inputs_shape = np.zeros((inputs.shape[0], 2))
        #     for i in range(inputs.shape[0]):
        #         inputs_shape[i, 0] = inputs.shape[2]
        #         inputs_shape[i, 1] = inputs.shape[3]
        #
        #     anchors_tmp = [[220, 125, 128, 222, 264, 266],
        #                     [35, 87, 102, 96, 60, 170],
        #                     [10, 15, 24, 36, 72, 42] ]
        #     bbox, bbox_num = self.post_process(
        #         out, anchors_tmp,
        #         paddle.to_tensor(inputs_shape), 1.)
        #
        #     return bbox, bbox_num

class PPYoloMobileNetV3(paddle.nn.Layer):
    """PPYolo. based MobileNetV3"""

    def __init__(self):
        super(PPYoloMobileNetV3, self).__init__()
        self.backbone = mobilenet_v3.MobileNetV3(model_name="small", scale=1.0, with_extra_blocks=False,
                                                 extra_block_filters=[], feature_maps=[9, 12])
        # self.neck = yolo_fpn.PPYOLOFPN(in_channels=[96, 304], coord_conv=True, conv_block_num=0,
        #                                spp=True, drop_block=True)
        self.neck = yolo_fpn.PPYOLOFPN(in_channels=[48, 96], coord_conv=True, conv_block_num=0,
                                       spp=True, drop_block=True)
        self.yolo_head = yolo_head.YOLOv3Head(in_channels=[512, 256],
                                              anchors=[[11, 18], [34, 47], [51, 126],
                                                       [115, 71], [120, 195], [254, 235]],
                                              anchor_masks=[[3, 4, 5], [0, 1, 2]],
                                              num_classes=3,
                                              loss=yolo_loss.YOLOv3Loss(num_classes=3, ignore_thresh=0.5,
                                                                        label_smooth=False, downsample=[32, 16],
                                                                        scale_x_y=1.05, iou_loss=iou_loss.IouLoss()))

        self.post_process = post_process.BBoxPostProcess(decode=YOLOBox(num_classes=3, conf_thresh=0.005,
                                                                        downsample_ratio=32, clip_bbox=True,
                                                                        scale_x_y=1.05), num_classes=3,
                                                         nms=MultiClassNMS(keep_top_k=100, nms_threshold=0.45,
                                                                           nms_top_k=1000, score_threshold=0.005))

    def forward(self, inputs, targets=None):
        """
        net forward
        :param inputs: inputs
        :param targets: labels
        :return:
        """
        out = self.backbone(inputs)
        out = self.neck(out)
        # out = self.head(out, inputs)
        if targets is not None:
            return self.yolo_head(out, targets)
        else:
            yolo_head_outs = self.yolo_head(out)
            inputs_shape = np.zeros((inputs.shape[0], 2))
            for i in range(inputs.shape[0]):
                inputs_shape[i, 0] = inputs.shape[2]
                inputs_shape[i, 1] = inputs.shape[3]

            bbox, bbox_num = self.post_process(
                yolo_head_outs, self.yolo_head.mask_anchors,
                paddle.to_tensor(inputs_shape), 1.)

            return bbox, bbox_num

        return out


if __name__ == '__main__':
    net = PPYoloTiny()
    # layer_state_dict = paddle.load('./MobileNetV3_small_x1_0_ssld_pretrained.pdparams')
    # layer_state_dict = paddle.load('./ppyolo_mbv3_small_coco.pdparams')
    # net.set_state_dict(layer_state_dict)
    paddle.summary(net, (1, 1, 160, 320))
    paddle.flops(net, (1, 1, 160, 320))

    # paddle.summary(net, (1, 3, 128, 224))
