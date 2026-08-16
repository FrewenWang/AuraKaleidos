# 可复现训练说明

本文档描述不依赖百度内部 AFS、私有标注或 NVIDIA GPU 的训练路径。它保留原工程的
`PPYoloTiny`、MobileNetV3、PP-YOLO FPN 和 YOLOv3 loss，只替换数据准备、训练调度和
推理解码部分。

## 数据策略

`scripts/prepare_dataset.py` 首选 [scikit-learn Olivetti Faces](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html)
提供的人脸数据（400 张
64×64 灰度人脸），按身份切成 80% 训练前景、20% 验证前景，再把 1～3 张人脸确定性地
合成到 96×160 的随机低频背景中。每次使用相同 seed 都会生成相同的图片和检测框。

这套数据只用于验证代码、损失下降、模型保存和推理链路。它的背景与构图是合成的，不能
代表真实道路/舱内人脸分布；部署模型应替换为 WIDER FACE 或经过授权的业务数据。

## 创建环境

在工程根目录 `AliceBaiduFaceDetection` 执行：

```bash
conda env create -f environment.yml
conda activate alice-face-paddle
python -c "import paddle; paddle.utils.run_check()"
```

环境固定 Python 3.10 和 PaddlePaddle 2.6.2。根据
[PaddlePaddle macOS 安装说明](https://www.paddlepaddle.org.cn/documentation/docs/en/2.6/install/pip/macos-pip_en.html)，
macOS 只能使用 CPU；Linux/Windows
也可直接使用 CPU 配置。若在受支持的 NVIDIA Linux 环境安装 `paddlepaddle-gpu`，可把
配置中的 `training.device` 改为 `gpu`。

## 生成数据并训练

```bash
python scripts/prepare_dataset.py --config configs/repro_cpu.yaml
python scripts/train_repro.py --config configs/repro_cpu.yaml
```

重新生成数据时使用 `--force`。快速冒烟测试可使用 `--epochs 1`。默认输出：

```text
outputs/repro_cpu/
├── best.pdparams     # 验证 loss 最低的模型
├── last.pdparams     # 最后一轮模型
├── best.pdstate      # 最佳轮完整训练状态
├── last.pdstate      # 最后一轮完整训练状态
├── config.yaml       # 本次训练配置快照
└── metrics.json      # loss 历史和合成验证集指标
```

继续训练：

```bash
python scripts/train_repro.py --resume outputs/repro_cpu/last.pdstate --epochs 4
```

`.pdstate` 会恢复模型、优化器、epoch、最佳 loss、历史记录以及可序列化的随机状态；加载
旧 `.pdparams` 时会自动降级为只恢复权重。`--epochs` 表示本次继续执行的轮数。训练器还会
根据全局 seed 和 epoch 生成确定性采样/增强 seed，使断点恢复后的后续 epoch 保持一致。

## 推理

```bash
python scripts/predict_repro.py data/olivetti_synthetic/images/val/00000.png
```

结果保存为 `outputs/repro_cpu/prediction.png`。模型输入沿用历史工程的 3 通道归一化；
灰度图片会复制到三个通道。

扫描不同置信度阈值：

```bash
python scripts/evaluate_repro.py --thresholds 0.005,0.01,0.02,0.05,0.1,0.15
```

## 自动检查

```bash
python -m unittest discover -s tests -v
```

检查覆盖任意图片尺寸、非法框、空标注、stretch/letterbox 坐标转换、WIDER 标注转换、
IoU、AP、YOLO target、模型前反向传播、原生 Paddle 解码/NMS 和完整 checkpoint 恢复。
完整训练是耗时集成测试，不放入普通单测。GitHub Actions 会在 Linux、Windows 执行模型
测试，并在 macOS 检查源码和包元数据。

## 使用真实数据

训练清单是 JSON Lines，每行格式如下：

```json
{"image": "images/train/00000.png", "boxes": [[20, 10, 55, 60]]}
```

坐标为像素级 `[x1, y1, x2, y2]`，图片路径相对于清单目录。准备好真实图片后生成
`train.jsonl` 和 `val.jsonl`，并在配置中把 `data.root` 指向该目录即可。清单记录原图坐标；
加载器会清理 NaN、越界和无面积框，并让图片与框同步执行缩放、letterbox 和水平翻转。
推理结果也会自动映射回原始图片坐标。没有人脸的负样本使用空列表 `"boxes": []`。
`data.max_boxes` 控制每张图片参与训练的最大框数；合成集默认 10，WIDER 配置提高到 512，
避免多脸场景被历史固定长度静默截断。

### WIDER FACE

下载器使用 WIDER FACE 官方页面给出的数据镜像和标注地址，固定 SHA-256，校验 ZIP 后才
解压。它只依赖 Python 标准库，因此可以在安装 PaddlePaddle 之前运行：

```bash
python scripts/download_wider_face.py --dataset-root data/wider
```

下载约 1.8 GB，解压后的目录为：

```text
data/wider/
├── WIDER_train/images/
├── WIDER_val/images/
└── wider_face_split/
    ├── wider_face_train_bbx_gt.txt
    └── wider_face_val_bbx_gt.txt
```

然后执行：

```bash
python scripts/prepare_wider_face.py --dataset-root data/wider
python scripts/fit_anchors.py --config configs/wider_small_faces.yaml \
  --output outputs/wider_small_faces/anchors.json
python scripts/train_repro.py --config configs/wider_smoke.yaml
# 冒烟通过且具备足够训练时间或 GPU 后：
python scripts/train_repro.py --config configs/wider_small_faces.yaml
```

转换器会过滤官方标注中 `invalid=1` 或无面积的框，并生成
`data/wider/manifests/{train,val}.jsonl`。当前 `AP50/mAP50:95` 是工程通用指标，不等同于
WIDER FACE Easy/Medium/Hard 官方协议；正式对外报告前仍需接入官方 difficulty split。
`fit_anchors.py` 会按配置中的输入尺寸和 resize 模式统计真实训练框，输出当前 Anchor 与
聚类 Anchor 的 mean best IoU、Recall@0.5 和 Recall@0.75，结果写入
`outputs/wider_cpu/anchors.json`。确认结果后再把建议值写入配置并启动新实验。
本次 `320×192 + letterbox` 聚类得到 `[[1,1], [2,2], [2,3], [3,4], [4,6],
[6,8], [10,12], [15,20], [33,44]]`，训练框 Recall@0.5 从旧 Anchor 的 `0.2325`
提高到 `0.9748`。`wider_smoke.yaml` 会确定性抽取训练/验证图片，并通过
`max_steps_per_epoch` 将训练限制为 8 个 batch，用于快速验证真实数据链路；
`wider_cpu.yaml` 使用完整数据，但只适合作为低分辨率 CPU 基线。

统计全部有效框后，320×192 输入中的人脸短边中位数仅 `4.07 px`，只有 `24.43%` 达到
`8 px`。因此正式训练增加 `wider_small_faces.yaml`：输入为 640×384、batch size 为 2，
聚类 Anchor 为 `[[2,3], [3,4], [4,6], [6,8], [9,11], [12,16], [19,24],
[30,39], [66,88]]`，156,994 个框上的 mean best IoU 为 `0.7759`、Recall@0.75 为
`0.6477`。该配置 CPU 也能运行，但建议在 NVIDIA Linux 环境把 `training.device` 改为
`gpu` 并按显存调整 batch size。

完整训练配置允许每图 512 个框；原始训练集中最大值为 1962，极端拥挤图片仍会截断，
应在后续实验中结合显存调整或使用拥挤区域裁剪策略。
可以使用 `--max-steps` 临时覆盖配置中的 batch 上限，而不修改 YAML。

### 本机 WIDER FACE 链路基线

2026-08-08 下载与转换结果如下；`data/` 与 `outputs/` 均由 Git 忽略：

| 数据 | 图片 | 有效人脸 | 过滤 invalid 框 |
|---|---:|---:|---:|
| train | 12,880 | 156,994 | 2,426 |
| val | 3,226 | 39,112 | 596 |

320×192 冒烟配置从合成模型权重迁移后，在固定抽取的 256 张训练图上继续训练 8 轮，
验证 loss 从 `921.97` 降至最低 `154.74`；32 张验证图上的 AP50 为 `0.00012`。
这个结果证明真实数据下载、转换、训练、断点恢复和评估链路已经跑通，但不代表模型已经
收敛。低 AP 与极小目标、训练轮数不足以及通用 AP（非 WIDER 官方协议）有关，后续精度
实验应从 640×384 配置开始。

## 几何与模型配置

`data.resize_mode` 支持：

- `stretch`：直接缩放到目标宽高，兼容既有合成基线和权重。
- `letterbox`：保持长宽比并居中填充，推荐用于真实图片。

`model.anchors`、`model.anchor_masks` 和 `model.num_classes` 是 target、模型 Head、解码和
评估共同使用的唯一配置来源。启动时会检查 Anchor 尺寸、Mask 数量和索引范围，避免训练
与推理静默使用不同参数。当前可复现流程限定为单类人脸检测。

## 本机基线结果

2026-08-07 在 Intel macOS 13（CPU）上先训练 8 轮，再从最佳权重续训 12 轮。最终最佳
权重在 60 张合成验证图、IoU 0.5、置信度 0.20 下得到：

| 指标 | 结果 |
|---|---:|
| 最佳验证 loss | 24.8575 |
| Precision@0.5 | 0.7660 |
| Recall@0.5 | 0.5625 |
| F1@0.5 | 0.6486 |
| AP50 | 0.6108 |
| mAP50:95 | 0.2054 |

这是合成域基线，只证明训练、保存和推理流程有效，不代表 WIDER FACE 或真实业务数据精度。
新的评估入口还会从 `ap_confidence_floor` 开始收集预测，并报告 AP50 和 mAP50:95；固定
阈值的 Precision/Recall/F1 用于选择部署工作点，AP 用于比较不同训练版本。

## 导出 Paddle 推理模型

```bash
python scripts/export_repro.py \
  --verify-image data/olivetti_synthetic/images/val/00001.png
```

导出结果位于 `outputs/repro_cpu/export/model.pdmodel` 和 `model.pdiparams`。导出模型返回
三个尺度的原始 YOLO Head Tensor，解码与 NMS 由调用侧执行；脚本会自动比较动态图和
静态图输出，最大绝对误差超过 `1e-5` 时判定失败。

## 导出和验证 ONNX

Paddle 2.6 模型使用与之兼容的 `paddle2onnx==1.0.9`；较新的 Paddle2ONNX 2.x 要求
Paddle 3.x，不应直接升级当前训练环境。依次执行：

```bash
paddle2onnx --model_dir outputs/repro_cpu/export --model_filename model.pdmodel --params_filename model.pdiparams --save_file outputs/repro_cpu/export/model.onnx --opset_version 11 --enable_onnx_checker True
python scripts/verify_onnx.py --verify-image data/olivetti_synthetic/images/val/00001.png
```

导出的 ONNX 同样返回三个原始 YOLO Head Tensor。验证脚本使用 ONNX Runtime CPU 后端，
逐个比较 Paddle 动态图输出，默认最大绝对误差上限为 `1e-4`。命令本身不依赖 Bash 专有
语法，可在 Windows PowerShell、macOS 和 Linux 终端中直接使用。

## 历史代码与兼容层

- `train_face_local.py` 等入口保留用于考古，但包含已失效的本机/AFS 路径、剪枝模型路径和
  旧 PaddleCloud 假设，不作为复现入口。
- Paddle 2.6 移除了 `paddle.fluid`；`ops.py`、`layers.py` 使用 `paddle.base` 兼容导入。
- 历史推理后处理调用私有的 `core.ops`。现已改为公开的 `paddle.vision.ops`；可复现脚本
  也能从 YOLO head 读取原始输出，用 NumPy/OpenCV 解码和 NMS，避免依赖 Paddle 私有 ABI。
- 历史网络把名为 `hard_swish` 的分支实际实现为普通 Swish。为了兼容 Paddle2ONNX，代码
  使用数学等价的 `x * sigmoid(x)`，没有改变已有权重的推理语义。
- 历史脚本曾包含明文 PaddleCloud/AFS 凭据，现已改为环境变量或占位符。历史值仍可能
  存在于 Git 提交记录中，仓库管理员应在相应平台立即轮换并吊销旧凭据。
