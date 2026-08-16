# Alice 灰度人脸检测

这是一个基于 PaddlePaddle 的轻量级单类人脸检测工程，主体模型为 MobileNetV3 +
PP-YOLO Tiny。仓库保留了历史百度云训练代码，同时提供可在普通 CPU 开发机上独立复现
的数据准备、训练、评估、推理和模型导出流程。

## 工程结构

```text
AliceBaiduFaceDetection/
├── src/alice_face_detection/ # 模型、数据集和公共 Python 模块
├── scripts/                  # 推荐命令行入口
│   ├── evaluation/           # 历史评估工具
│   └── legacy/               # 依赖旧环境的训练/推理入口
├── configs/                  # 训练和推理配置
├── data/                     # 数据集及生成数据（Git 忽略）
├── assets/                   # 随代码管理的图片、字体等静态素材
├── logs/                     # 运行日志
├── outputs/                  # 权重、指标和导出模型（Git 忽略）
├── tests/                    # 单元与模型回归测试
├── docs/                     # 详细文档
├── pyproject.toml            # Python 包元数据
└── environment.yml           # 固定版本的 Conda 环境
```

根目录下的源码已整理为 `alice_face_detection` 包。`scripts/legacy/` 中的脚本包含历史个人
数据路径或云服务约定，不作为新实验入口；推荐流程只使用 `scripts/` 根目录下带 `_repro`
后缀的入口。

## 快速开始

以下命令在 Windows、macOS 和 Linux 上一致；Windows 请在 Anaconda Prompt 或
PowerShell 中执行。

```bash
cd AliceBaiduFaceDetection
conda env create -f environment.yml
conda activate alice-face-paddle
python -m pip install -e . --no-deps
python scripts/prepare_dataset.py --config configs/repro_cpu.yaml
python scripts/train_repro.py --config configs/repro_cpu.yaml
python scripts/predict_repro.py data/olivetti_synthetic/images/val/00000.png
```

脚本自身也会加载本地 `src/`，因此开发阶段即使未执行可编辑安装也能直接运行。默认数据
由公开 Olivetti 人脸前景与程序生成的背景合成，仅用于验证训练闭环，不等价于真实道路或
车舱场景数据。训练产物写入 `outputs/repro_cpu/`，训练日志写入
`logs/repro_cpu/train.log`。

图片和标注框会使用同一套 `stretch` 或 `letterbox` 几何变换，因此训练清单可以直接记录
原始图片坐标。推荐先用合成配置验证流程，再用 `configs/wider_smoke.yaml` 验证真实数据
链路。正式训练优先使用 640×384 的 `configs/wider_small_faces.yaml`；320×192 的
`configs/wider_cpu.yaml` 更节省算力，但会丢失大量小脸信息。

## 常用命令

```bash
# 运行回归测试
python -m unittest discover -s tests -v

# 从完整 checkpoint 继续训练 4 轮
python scripts/train_repro.py --resume outputs/repro_cpu/last.pdstate --epochs 4

# 评估固定阈值、AP50 和 mAP50:95
python scripts/evaluate_repro.py

# 下载、校验并解压 WIDER FACE（约 1.8 GB）
python scripts/download_wider_face.py --dataset-root data/wider

# 转换 WIDER FACE 标注
python scripts/prepare_wider_face.py --dataset-root data/wider

# 快速验证和正式小目标训练
python scripts/train_repro.py --config configs/wider_smoke.yaml
python scripts/train_repro.py --config configs/wider_small_faces.yaml

# 导出 Paddle 静态模型
python scripts/export_repro.py

# 转换并验证 ONNX
paddle2onnx --model_dir outputs/repro_cpu/export --model_filename model.pdmodel --params_filename model.pdiparams --save_file outputs/repro_cpu/export/model.onnx --opset_version 11 --enable_onnx_checker True
python scripts/verify_onnx.py
```

Intel macOS CPU 合成验证集现有基线：Precision `0.7660`、Recall `0.5625`、F1
`0.6486`、AP50 `0.6108`、mAP50:95 `0.2054`。这些结果仅用于验证工程闭环和比较
代码版本。

数据来源、清单格式、指标定义、checkpoint 和导出约束详见
[可复现训练文档](docs/reproducible-training.md)。
