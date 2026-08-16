# PyTorch 2.0 学习示例

按课程章节整理的 PyTorch 教学代码，覆盖张量与 Softmax、手写神经网络、卷积/MNIST、
ResNet、词向量、循环网络、Transformer、强化学习、语音和人脸识别。

## 使用方式

```bash
cd FantasyPython/alice-pytorch
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install torch torchvision numpy matplotlib
python 01_hello_pytorch.py
python 02_pytorch_test_gpu.py
```

每章依赖不同，部分示例还需要 pandas、Gym、音频/视觉数据集或 GPU。请先阅读目标脚本，
不要把全部章节视为一个可一次运行的工程。

`docs/` 中的 PPTX 是课程资料；目录内历史 `.pth` 文件是示例权重。新增大文件应使用外部
存储，并在文档中记录来源、版本和校验值，不提交到 Git 仓库。
