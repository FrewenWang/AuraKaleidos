# FantasyAI

模型学习、训练、视觉处理与推理示例集。这里是按框架/任务分组的实验室，不是一个可安装 Python 包，也没有全局通用的依赖版本组合。

## 目录地图

| 目录 | 主要内容 | 关键依赖/输入 |
|---|---|---|
| `AliceAILearn/` | AI 基础教程与 Notebook | Jupyter, NumPy |
| `AliceTensorFlow/`, `AliceTensorflow2.0/` | TensorFlow 1.x/2.x 练习 | TensorFlow，部分示例需数据集 |
| `AlicePyTorch/` | PyTorch 基础、分类、目标检测 | torch, torchvision, 权重/数据集 |
| `AlicePaddlePaddle/` | PaddlePaddle 算子与训练实验 | paddlepaddle |
| `AliceKaleidoYolo/` | YOLO v3/v5/v8/v11 相关代码 | 各版本各自的 requirements 与权重 |
| `AliceBaiduFaceDetection/` | 人脸框/关键点训练实验 | PaddlePaddle, OpenCV, 私有数据 |
| `AliceInference/` | ONNX/视觉模型推理 | onnxruntime/OpenCV，模型和样例图 |
| `AliceOpenCV/` | Python/C++ OpenCV 演示 | OpenCV，C++ 部分使用 CMake |
| `AliceModelConvert/` | 模型导出/转换草稿 | ONNX 及目标框架 |
| `AliceAutoDriving/` | 自动驾驶学习索引 | 视具体文件而定 |

每个子工程现在都有根 `README.md`。统一目录规则见 [`docs/directory-conventions.md`](docs/directory-conventions.md)，逐项盘点和保留例外见 [`docs/subprojects.md`](docs/subprojects.md)。

## 目录约定

```text
<subproject>/
├── README.md       # 必须保留在子工程根目录
├── docs/           # 架构、原理、数据格式等扩展文档
├── src/            # 可复用源码/包
├── examples/       # demo 和 Notebook
├── tests/          # 有断言的自动测试
├── configs/        # 非敏感配置
├── scripts/        # 下载/训练/转换/部署辅助脚本
└── assets/         # 小型文档/demo 资源
```

训练产物使用 `outputs/`/`runs/`/`checkpoints/`，数据集使用本地 `datasets/`；它们默认不进入 Git。对于 YOLO 等上游副本，保留上游布局，不为了形式统一而破坏包导入。

## 使用方式

1. 先选中一个子目录，阅读其 README/脚本头部，不要在同一环境中同时安装所有框架。
2. 对该实验建立独立虚拟环境，尤其是不同 YOLO/TensorFlow/PaddlePaddle 版本。
3. 通过命令行参数或环境变量传入模型、图片和数据集路径。历史脚本中的绝对路径只是作者环境样例，不可直接复用。

例如，YOLOv8 有自己的依赖文件：

```bash
python3 -m venv .venv-yolov8
source .venv-yolov8/bin/activate       # Windows: .venv-yolov8\Scripts\activate
python -m pip install -r AliceKaleidoYolo/yolo-v8/requirements.txt
```

## 验证和平台边界

```bash
# FantasyAI 专用结构、Markdown 位置和 Python 语法检查
cd FantasyAI
python tools/check_structure.py
```

- CPU 通常可运行教程和小型推理；CUDA 需 NVIDIA 驱动与匹配的 PyTorch/TensorFlow。
- macOS 可用 CPU，部分 PyTorch 示例可用 MPS；TensorFlow GPU 与 CUDA 代码不能原样使用。
- Windows 的中文路径需确保终端、Python 和数据集均使用 UTF-8。
- 当前自动检查不会下载权重、启动摄像头、打开 GUI 或执行长时间训练。

已知技术债务是部分训练/推理脚本仍硬编码本机数据路径，且子工程依赖未完全锁定。新代码应优先使用 `argparse` + `pathlib` 收敛这些差异。
