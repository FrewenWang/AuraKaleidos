# Alice Auto Driving

自动驾驶数学基础工具，当前包含欧氏距离/马氏距离关联和二次曲线 RANSAC。

- `src/alice_auto_driving/`：可复用实现。
- `tests/`：无图形界面、可重复的单元测试。
- `examples/`：绘图和交互演示。

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```
