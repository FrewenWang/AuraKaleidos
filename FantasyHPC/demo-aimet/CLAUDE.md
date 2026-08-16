# demo-aimet

AIMET 的端到端上手示例，覆盖 ONNX 与 PyTorch 两条量化路径，以及安装环境的快速检查。

## 脚本与说明

- `aimet_onnx_quickstart.py`：基于 ONNX 模型的 AIMET 快速入门。
- `aimet_pytorch_quickstart.py`：基于 PyTorch 模型的 AIMET 快速入门。
- `check_aimet_installation.py`：检查 AIMET 安装与环境是否正确。
- `aimet_mobilenet_quantization.py`：MobileNetV2 量化示例。
- `03_demo_aimet_mobilenet_quant_pytorch.py`：以 MobileNet 为例的 PyTorch 量化完整流程。

## 运行方式

脚本均为独立 Python 脚本，数字前缀为建议运行顺序：

```bash
python aimet_pytorch_quickstart.py   # 或先跑 ONNX 版
python check_aimet_installation.py
python 03_demo_aimet_mobilenet_quant_pytorch.py
```

## 依赖

依赖 `aimet_torch` / `aimet_onnx`、`torch` 等，本目录无统一 `requirements.txt`，需保证运行环境已安装 AIMET。

## 编辑注意

- 脚本主要用于验证与演示，路径/参数多为示意值，复用于实际模型前需适配。
- 本仓库文档以中文为主，新增注释与文档字符串请使用中文。
