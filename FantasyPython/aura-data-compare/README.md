# Aura Data Compare

数值数组误差比较工具。核心实现位于 `src/`，数据生成与可视化命令位于 `scripts/`，手动报告示例位于 `examples/`。

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

测试不打开图形窗口；生成数据和图表写入工程的 `data/` 或 `outputs/`，这些运行产物默认不提交。
