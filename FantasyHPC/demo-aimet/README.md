# demo-aimet

AIMET 端到端快速入门，包含 ONNX、PyTorch 量化模拟、环境检查和 MobileNet 量化示例。

```bash
cd FantasyHPC/demo-aimet
python check_aimet_installation.py
python aimet_pytorch_quickstart.py
# 或 ONNX 路径：
python aimet_onnx_quickstart.py
python 03_demo_aimet_mobilenet_quant_pytorch.py
```

需要与宿主 PyTorch/CUDA 匹配的 `aimet_torch` 或 `aimet_onnx`。不同 AIMET 发行版的 API
可能不兼容，建议为本目录创建独立环境。脚本是教学实验，可能下载模型或数据；运行前先
检查输入、输出和网络访问行为。
