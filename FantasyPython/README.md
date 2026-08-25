# FantasyPython

Python 教程、数据处理、视觉实验和设备桥接工具集。每个目录都是独立 demo，目前没有共享包、统一 CLI 或统一依赖文件。

| 目录 | 内容 | 常见依赖 |
|---|---|---|
| `alice_examples/` | Python/OpenCV 基础脚本 | Python, OpenCV |
| `ailice-matplotlib/` | Matplotlib 绘图 | matplotlib, numpy |
| `alice-auto-driving/` | RANSAC、马氏距离关联可视化 | numpy, scipy, matplotlib |
| `alice-pyopencv/` | OpenCV 图像/视频处理 | opencv-python, numpy |
| `aura-data-compare/` | raw/text 数据比对 | numpy，视脚本而定 |
| `aura-midb/` | Android/Mi 调试工具环境 | Bash/Zsh, adb |
| `aura-qdb/` | QNX 调试工具环境 | Bash/Zsh, ssh/scp |
| `aura-pyutils/` | 待扩展 Python 工具 | 无稳定 API |

## 运行原则

```bash
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install <当前 demo 需要的依赖>
python <目标脚本.py>
```

- 可视化脚本可能调用 `plt.show()`，CI/无桌面 Linux 需设置 `MPLBACKEND=Agg` 或将结果保存为图片。
- GPU 脚本应先检查 CUDA/MPS，不要默认 `.cuda()` 一定可用。
- 历史示例中存在本机绝对路径；运行前将它们替换为命令行参数或相对于 `Path(__file__)` 的路径。
- `zh.tsv`、`.pth` 等大文件不纳入 Git，应由外部数据存储管理。

## 验证

```bash
# 有 tests/ 的标准化子工程可直接运行 pytest，无需先 editable 安装
cd FantasyPython/<subproject>
python -m pytest -q

# 无测试的历史 demo 只检查目标脚本
python -m py_compile <目标脚本.py>
```

`test_ransac.py` 和 `test_mahalanobis_*.py` 是交互式演示，不是自动化回归测试；它们会打开绘图窗口。
