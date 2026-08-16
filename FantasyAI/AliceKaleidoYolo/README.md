# AliceKaleidoYolo

YOLO 不同世代的学习、训练和推理工程。

| 目录 | 定位 | 结构策略 |
|---|---|---|
| `yolo-v3/` | 自包含 PyTorch YOLOv3 | 保留原有 CLI 和 `utils/` |
| `yolo-v8/` | YOLOv5/Ultralytics 风格历史副本 | 不重排上游 `src/`，本地脚本放 `scripts/` |
| `yolo-v11/` | Ultralytics YOLO11 安装/压缩笔记 | 上游包与本地文档分离 |
| `datasets/` | 历史数据目录 | 大数据不应由 Git 跟踪 |

每个版本必须使用独立虚拟环境。`test.py` 是模型评估 CLI，不是 pytest 单元测试。权重、COCO 数据、`runs/` 和导出模型应放外部存储，不提交到 Git 仓库。
