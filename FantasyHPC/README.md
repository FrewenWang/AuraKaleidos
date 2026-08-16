# FantasyHPC

模型量化、转换、推理和边缘部署示例集。目录名保留了历史命名，实际职责更接近 Model Optimization & Deployment。

## 处理链路

```text
PyTorch 模型
   ├── AIMET 量化/量化感知训练  alice-aimet, demo-aimet
   └── ONNX 导出                  aura-model-converter/onnx
          ├── TensorRT engine       aura-model-converter/tensorrt, aura-tensorrt
          └── QNN 转换/量化         aura-model-converter/qnn*

任务示例：YOLOv3 / 蒸馏 / Flask 服务
Jetson 运行时：alice-jetson-inference
```

| 目录 | 用途 | 运行前提 |
|---|---|---|
| `alice-aimet/`, `demo-aimet/` | AIMET 量化实验 | 匹配的 Linux/Python/PyTorch/AIMET |
| `aura-model-converter/` | ONNX、TensorRT、QNN 转换 | 对应 SDK；用户提供模型和校准数据 |
| `aura-tensorrt/` | TensorRT Python API 验证 | NVIDIA GPU + CUDA + TensorRT |
| `aura-object-detection/` | YOLOv3 训练/检测/部署 | PyTorch + COCO 格式数据/权重 |
| `model-distillation/` | 教师-学生蒸馏 | PyTorch + 训练数据 |
| `model_deploy/` | Flask 推理服务示例 | 模型、Web 依赖，只适合演示 |
| `alice-jetson-inference/` | NVIDIA Jetson 推理引擎及示例 | Jetson/JetPack，CMake，上游代码快照 |
| `test/` | PyTorch CPU/MPS/CUDA 环境信息 | PyTorch；是 smoke check，不是断言测试 |

## 可移植的验证

```bash
# 已安装 PyTorch 的环境检查
cd FantasyHPC
python test/test_pytorch_cuda.py
```

Jetson 构建必须在支持的 Linux/JetPack 环境进行：

```bash
cd FantasyHPC
cmake -S alice-jetson-inference -B alice-jetson-inference/build
cmake --build alice-jetson-inference/build -j
```

macOS 可执行纯 Python/ONNX 操作和 MPS 检查，不能构建 Jetson/TensorRT。Windows 可执行部分 PyTorch/ONNX 示例，QNN/TensorRT 是否可用取决于 SDK 发行包。

## 必须先配置的项

历史 QNN/训练/部署脚本中仍有 `/home/...`、`/Users/...` 和 Windows 盘符路径。运行前应改为环境变量或参数，至少包括 `QNN_SDK_ROOT`、模型路径、输入列表和输出目录。不要在普通开发机上执行可能产生大量输出、需要许可 SDK 或依赖硬件的流程。

`alice-jetson-inference` 是普通 Git 文件形式保存的上游代码快照，不是本仓库 submodule。
当前快照缺少部分示例和数据文件，上游 README 中有一些相对链接无法在本地解析；需要完整
教程时应对照 `dusty-nv/jetson-inference` 上游版本。它还包含历史 Python 2 工具，自有
Python 语法检查不扫描该目录，应按上游支持矩阵单独验证。
