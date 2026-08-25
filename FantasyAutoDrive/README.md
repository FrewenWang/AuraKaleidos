# FantasyAutoDrive

自动驾驶算法学习集合。可复用实现进入各子工程的 `src/`，可直接运行的教学程序进入 `examples/`，
模型、数据和运行结果分别放入 `models/`、`data/` 与 `outputs/`。

| 子工程 | 内容 |
| --- | --- |
| `covariance` | 协方差示例 |
| `kalman_filter` | 卡尔曼滤波推导与示例 |
| `kalman_filter_with_yolo11_objects_tracker` | YOLO + 卡尔曼目标跟踪完整工程 |
| `line_smooth` | 二次规划路径平滑实现 |

## 可自动验证的工程

YOLO + 卡尔曼跟踪子工程已采用标准 `src/` 包布局，并提供不下载模型、不打开摄像头的测试：

```bash
cd FantasyAutoDrive/kalman_filter_with_yolo11_objects_tracker
python -m pip install -e '.[dev]'
python -m pytest -q
```

其他目录主要是教学脚本，可能打开 Matplotlib 窗口或读取本地数据。无桌面 CI 应设置
`MPLBACKEND=Agg`，并只执行已经具有断言和固定 fixture 的测试。
