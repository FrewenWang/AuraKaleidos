# Scripts

- 根目录下的 `*_repro.py` 是推荐使用的跨平台数据、训练、评估、推理与导出入口。
- `download_wider_face.py` 下载 WIDER FACE train/val 和官方标注，并校验 SHA-256 与 ZIP。
- `prepare_wider_face.py` 把官方 WIDER FACE train/val 标注转换为工程 JSONL 清单。
- `fit_anchors.py` 根据训练清单和目标输入尺寸聚类 Anchor，并报告覆盖率对比。
- `evaluation/` 保存历史 WIDER FACE 和结果检查工具。
- `legacy/` 保存依赖旧百度云环境、硬编码数据路径或专用设备的历史入口，仅供参考。

推荐脚本可以直接从项目根目录运行。历史 Python 脚本如需尝试，请先设置
`PYTHONPATH=src`，并自行替换其中已失效的外部路径。
