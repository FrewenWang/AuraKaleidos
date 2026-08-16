# PyTorch 人脸框训练示例

## 结构

```text
face_rect_train/
├── README.md
├── src/                  # Dataset、YOLO MobileNet 模型和训练草稿
└── tools/                # WIDER FACE 数据下载脚本
```

WIDER FACE 数据集：[WIDER FACE 官网](http://shuoyang1213.me/WIDERFACE/)。下载脚本位于 `tools/download_wider_face_datasets.sh`，执行前请检查下载地址与目标目录。

`src/` 已作为 Python 包处理，包内使用相对导入。`train.py` 目前仍是实验草稿，包含占位数据路径，在补齐 CLI 配置、标注格式和小型 smoke test 前不应用于长时间训练。
