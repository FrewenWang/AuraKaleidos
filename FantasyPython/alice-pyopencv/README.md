# alice-pyopencv

Python OpenCV 学习工程。可复用函数位于 `src/alice_pyopencv/`，可运行代码位于 `examples/`，
Notebook 位于 `notebooks/`，图片和小型测试数据分别位于 `assets/` 与 `data/`。

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

示例程序应从工程根目录运行，并通过 `Path(__file__)` 解析资源，避免依赖当前工作目录。
