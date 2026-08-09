# FantasyCuda

CUDA 教学示例集合。源码统一放在 `src/`，Notebook 放在 `notebooks/`，本地构建结果进入
`build/`，不提交可执行文件。

## 构建

```bash
cmake -S . -B build
cmake --build build --parallel
```

没有 CUDA Toolkit（例如多数 macOS 环境）时，CMake 会正常完成配置并跳过 CUDA 目标。
