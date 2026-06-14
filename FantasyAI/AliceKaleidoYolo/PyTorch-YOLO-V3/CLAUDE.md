# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 项目概述

这是一个基于 PyTorch 实现的 YOLOv3 (You Only Look Once v3) 目标检测框架。该实现遵循 Darknet 原始配置格式，使用 `.cfg` 文件定义模型架构。

## 常用命令

### 训练
```bash
python train.py \
  --data_config config/coco.data \
  --pretrained_weights weights/darknet53.conv.74 \
  --epochs 100 \
  --batch_size 4 \
  --model_def config/yolov3.cfg  # 或 config/yolov3-tiny.cfg
```

### 评估（计算 mAP）
```bash
python test.py \
  --data_config config/coco.data \
  --weights_path weights/yolov3.weights \
  --iou_thres 0.5 \
  --conf_thres 0.001 \
  --nms_thres 0.5
```

### 图像推理检测
```bash
python detect.py \
  --image_folder data/samples \
  --weights_path weights/yolov3.weights \
  --conf_thres 0.8 \
  --nms_thres 0.4
```

### 下载预训练权重
```bash
bash scripts/download_weights.sh
```

## 架构说明

### 模型定义（配置驱动）
模型由 Darknet `.cfg` 文件定义在 `config/` 目录中。解析器 `utils/parse_config.py:parse_model_config()` 读取这些配置，`models.py:create_modules()` 动态构建 `nn.ModuleList`。关键层类型：
- `convolutional` → Conv2d + BatchNorm + LeakyReLU
- `maxpool` → MaxPool2d
- `upsample` → F.interpolate (nearest 模式)
- `route` → 指定层的拼接（特征金字塔）
- `shortcut` → 指定层的残差连接
- `yolo` → 该尺度下的检测层 YOLOLayer

### 模型架构 (`models.py`)
- **Darknet**: 主网络类，包含 `forward()`、`load_darknet_weights()`、`load_state_dict()`
- **YOLOLayer**: 检测层，预测边界框、目标置信度和类别概率
- **Upsample**: 使用 F.interpolate 的自定义上采样（nn.Upsample 已弃用）
- **EmptyLayer**: route/shortcut 连接的占位符

### 多尺度检测
YOLOv3 在 3 个尺度上进行检测。第一次检测发生在最后一个卷积层，然后将特征图上采样并与前面的层拼接进行第二次检测，重复此过程进行第三次检测。

### 损失计算
损失在 `models.py` 的 `Darknet.forward()` 中计算，包括：
- 目标置信度的二元交叉熵
- 类别预测的二元交叉熵
- 边界框坐标 (x, y, w, h) 的平方误差和

### 数据管道 (`utils/datasets.py`)
- **ListDataset**: 用于训练/验证（读取图像列表 + 标签）
- **ImageFolder**: 用于推理（读取文件夹中的图像）
- 图像会被填充为正方形、调整大小，并可选地进行增强（水平翻转）
- 标签格式：`[类别, x_center, y_center, width, height]`（归一化到 0-1）

### 目标构建 (`utils/utils.py:build_targets()`)
将真实框转换为训练目标：
- 基于 IoU 将每个真实框分配到最匹配的锚框
- 设置 `obj_mask`、`noobj_mask`、`tx`、`ty`、`tw`、`th`、`tcls` 张量
- 处理与物体重叠但不是最佳匹配的锚框的忽略阈值

### 非极大值抑制 (`utils/utils.py:non_max_suppression()`)
过滤检测结果：
1. 移除低于置信度阈值的预测
2. 对每个类别，按置信度排序并贪心抑制重叠框（IoU > nms_thres）
3. 返回每个检测 `[x1, y1, x2, y2, object_conf, class_score, class_pred]`

## 配置文件

### 模型配置 (`config/*.cfg`)
- `[net]` 部分：批次大小、图像尺寸、学习率调度
- `[yolo]` 部分：锚框掩码（使用哪些锚框）、类别数、抖动、阈值
- 锚框以逗号分隔的 width,height 对定义

### 数据配置 (`config/*.data`)
- `classes`: 类别数量
- `train`: 训练图像列表路径
- `valid`: 验证图像列表路径
- `names`: 类别名称文件路径
- `backup`: 检查点目录

## 重要注意事项

### 路径处理问题
`utils/datasets.py` 中包含硬编码的 Windows 路径（约第 85、96 行），在其他平台上会失败：
```python
img_path = 'E:\\eclipse-workspace\\PyTorch\\PyTorch-YOLOv3\\data\\coco' + img_path
label_path = 'E:\\eclipse-workspace\\PyTorch\\PyTorch-YOLOv3\\data\\coco\\labels' + label_path
```
需要修复以支持跨平台兼容性。

### 权重格式
代码库支持两种权重格式：
- `.weights`: Darknet 格式（通过 `model.load_darknet_weights()` 加载）
- `.pth`: PyTorch 状态字典（通过 `model.load_state_dict()` 加载）

### 依赖项
- PyTorch
- torchvision
- numpy
- matplotlib（用于检测可视化）
- terminaltables（用于训练日志）
- tqdm
- PIL/Pillow

## 关键超参数
- 默认图像大小：416×416
- 训练多尺度范围：320-448（步长 32）
- YOLOv3-tiny 锚框：10×14, 23×27, 37×58, 81×82, 135×169, 344×319
- YOLOv3 锚框：与 tiny 相同（每个尺度 3 个）
- 忽略阈值：0.7（IoU > 此值的锚框视为无物体）
