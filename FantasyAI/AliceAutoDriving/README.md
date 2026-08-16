# AliceAutoDriving

自动驾驶感知/拟合算法演示。当前只有 `lane_line/01.demo_ransac_regressor.py`，用 RANSAC 演示车道线回归。

```bash
python -m pip install numpy matplotlib scikit-learn
python lane_line/01.demo_ransac_regressor.py
```

该脚本是可视化 demo，不是单元测试。新的可执行演示放 `examples/<topic>/`，可复用算法放 `src/`，无 GUI 断言测试放 `tests/`。
