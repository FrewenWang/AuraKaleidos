# 如何运行本项目

本文档详细说明了如何从零开始运行这个 YOLOv3 PyTorch 实现。

## 前置要求

### 1. Python 版本
- Python 3.7 或更高版本

### 2. 安装依赖
```bash
pip install torch torchvision
pip install numpy matplotlib tqdm terminaltables pillow
```

**注意**: `utils/logger.py` 使用了 TensorFlow 进行日志记录。如果你不需要 TensorBoard 日志，可以忽略这个依赖，或者我们会在下面提供替代方案。

### 3. 下载预训练权重
```bash
bash scripts/download_weights.sh
```

这会在 `weights/` 目录下载以下文件：
- `yolov3.weights` - 完整 YOLOv3 模型权重
- `yolov3-tiny.weights` - 轻量级 YOLOv3-tiny 模型权重
- `darknet53.conv.74` - Darknet53 骨干网络权重（用于迁移学习）

## 数据准备

### COCO 数据集结构

项目期望的目录结构如下：
```
data/
└── coco/
    ├── images/
    │   ├── train2014/     # COCO 2014 训练图像 (118,287 张)
    │   └── val2014/       # COCO 2014 验证图像 (5,000 张)
    ├── labels/
    │   ├── train2014/     # 训练集标签 (.txt 文件)
    │   └── val2014/       # 验证集标签 (.txt 文件)
    ├── trainvalno5k.txt   # 训练列表
    ├── 5k.txt             # 验证列表
    └── coco.names         # 类别名称 (80 个类别)
```

### 数据准备步骤

#### 方法 1: 使用现有数据（推荐）

如果你的 `data/coco/` 目录已经有数据，只需运行准备脚本修复目录结构：

```bash
python scripts/prepare_data.py
```

#### 方法 2: 下载 COCO 数据集

1. 下载 COCO 2014 数据集：
   ```bash
   # 下载训练集图像 (18GB)
   wget http://images.cocodataset.org/zips/train2014.zip
   wget http://images.cocodataset.org/zips/val2014.zip

   # 解压到 images 目录
   unzip train2014.zip -d data/coco/images/
   unzip val2014.zip -d data/coco/images/
   ```

2. 下载标签文件：
   ```bash
   wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
   unzip annotations_trainval2014.zip -d data/coco/
   ```

3. 运行准备脚本：
   ```bash
   python scripts/prepare_data.py
   ```

#### 方法 3: 使用自定义数据集

如果要使用自己的数据集，需要：

1. 准备图像文件放在一个目录
2. 为每张图像创建对应的标签文件（YOLO 格式）：
   ```
   <类别ID> <x_center> <y_center> <width> <height>
   ```
   所有坐标值都是相对于图像尺寸的 0-1 之间的浮点数

3. 创建图像列表文件（每行一个图像路径）
4. 创建类别名称文件（每行一个类别名）
5. 创建数据配置文件 `config/custom.data`：
   ```
   classes=你的类别数
   train=path/to/train.txt
   valid=path/to/valid.txt
   names=path/to/classes.names
   backup=backup/
   eval=custom
   ```

## 运行命令

### 1. 测试推理（使用示例图像）

最简单的方式，使用预训练权重检测示例图像：

```bash
python detect.py \
  --image_folder data/samples \
  --weights_path weights/yolov3.weights \
  --class_path data/coco.names
```

检测结果会保存在 `output/` 目录。

### 2. 评估模型（计算 mAP）

```bash
python test.py \
  --model_def config/yolov3.cfg \
  --data_config config/coco.data \
  --weights_path weights/yolov3.weights \
  --batch_size 8 \
  --img_size 416
```

### 3. 训练模型

#### 从零开始训练（不推荐，需要大量时间和数据）：
```bash
python train.py \
  --model_def config/yolov3.cfg \
  --data_config config/coco.data \
  --epochs 100 \
  --batch_size 4 \
  --img_size 416
```

#### 使用预训练权重训练（推荐）：
```bash
python train.py \
  --model_def config/yolov3.cfg \
  --data_config config/coco.data \
  --pretrained_weights weights/darknet53.conv.74 \
  --epochs 100 \
  --batch_size 4 \
  --img_size 416
```

#### 使用 YOLOv3-tiny（更快，精度稍低）：
```bash
python train.py \
  --model_def config/yolov3-tiny.cfg \
  --data_config config/coco.data \
  --pretrained_weights weights/yolov3-tiny.weights \
  --epochs 100 \
  --batch_size 8 \
  --img_size 416
```

## 常见问题

### Q: 出现 "No module named tensorflow" 错误

`utils/logger.py` 使用 TensorFlow。如果你不需要 TensorBoard 日志，可以修改 `train.py`：

```python
# 注释掉这行
# from utils.logger import *

# 或者创建一个简单的替代 logger
class SimpleLogger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
    def scalar_summary(self, tag, value, step):
        print(f"[{step}] {tag}: {value}")
    def list_of_scalars_summary(self, tag_value_pairs, step):
        for tag, value in tag_value_pairs:
            print(f"[{step}] {tag}: {value}")
```

### Q: 出现 "FileNotFoundError" 数据路径错误

1. 检查数据目录结构是否正确
2. 运行 `python scripts/prepare_data.py` 修复目录
3. 检查 `config/coco.data` 中的路径是否正确

### Q: 出现 CUDA 内存不足

1. 减小 `--batch_size`（如 2 或 1）
2. 减小 `--img_size`（如 320）
3. 使用 YOLOv3-tiny 模型（`config/yolov3-tiny.cfg`）

### Q: 如何在 CPU 上运行

代码会自动检测 GPU 是否可用。如果没有 GPU，会自动使用 CPU，但速度会很慢。

### Q: 训练时如何查看日志

训练日志保存在 `logs/` 目录。如果使用 TensorFlow，可以用 TensorBoard 查看：

```bash
tensorboard --logdir=logs/
```

然后访问 `http://localhost:6006`

## 项目结构

```
PyTorch-YOLO-V3/
├── config/                 # 配置文件
│   ├── yolov3.cfg         # YOLOv3 模型配置
│   ├── yolov3-tiny.cfg    # YOLOv3-tiny 模型配置
│   ├── coco.data          # COCO 数据配置
│   └── custom.data        # 自定义数据配置模板
├── data/
│   ├── coco/              # COCO 数据集
│   └── samples/           # 示例图像
├── utils/                 # 工具函数
│   ├── datasets.py        # 数据集加载
│   ├── utils.py           # 通用工具（NMS, IoU 等）
│   ├── parse_config.py    # 配置解析
│   ├── logger.py          # 日志工具
│   └── augmentations.py   # 数据增强
├── scripts/               # 脚本
│   ├── download_weights.sh
│   └── prepare_data.py    # 数据准备脚本
├── train.py               # 训练脚本
├── test.py                # 评估脚本
├── detect.py              # 推理脚本
└── models.py              # 模型定义
```

## 学习资源

- [YOLOv3 论文](https://arxiv.org/abs/1804.02767)
- [Darknet 官网](https://pjreddie.com/darknet/yolo/)
- [PyTorch 官方教程](https://pytorch.org/tutorials/)
